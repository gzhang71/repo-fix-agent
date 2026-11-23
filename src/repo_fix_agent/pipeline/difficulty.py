"""Difficulty-aware data weighting from the student's own pass rate (free: no model calls).

  repo-fix eval runs/splits/train.jsonl --model student --samples 4 --temperature 0.7 --out runs/train/r1/student_eval.jsonl
  repo-fix difficulty-weights runs/train/r1/student_eval.jsonl --out runs/train/r1/weights.json \
      --pass-rates-out runs/train/r1/pass_rates.json [--teacher runs/train/r1/teacher_eval.jsonl]

Per instance, p = resolved samples / graded samples (errored samples are not counted). Routing:

  bucket      p                      teacher   goes to                         SFT weight
  gap         p <= low               solved    distill / on-policy relabel     high (1 - p)
  learning    low < p < high         any       RL (mixed groups), DPO pairs    1 - p
  mastered    p >= high              any       small replay only               floor
  unsolved    p <= low               failed    nothing yet (eval / later RL)   0

`low`/`high` default to 0 and 1, i.e. "never" and "always" over the samples taken. Teacher status
is optional; without it, p <= low instances count as `gap` (a correct trajectory may still exist
from engineer demos or distillation, and an instance with none simply produces no SFT record).

SFT weight = max(floor, 1 - p) x (0 if the teacher is known to fail and nothing else solved it).
Consumers:
  * sft-mix --instance-weights: multiplies each trajectory record's loss weight, then caps the
    total weight per instance (--max-weight-per-instance) so one instance can't dominate.
  * distill / onpolicy-distill --instance-weights --max-pass-rate: spend teacher calls only on
    instances the student doesn't already solve.
  * rl-rollouts --pass-rates: the curriculum keeps instances with mixed outcomes (GRPO only learns
    from groups that contain both wins and losses).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .sft_data import read_jsonl

BUCKETS = ("gap", "learning", "mastered", "unsolved")


def _key(rec: dict) -> str | None:
    meta = rec.get("metadata") or rec.get("meta") or {}
    k = meta.get("parent_id") or rec.get("instance_id") or meta.get("instance_id")
    return str(k) if k else None


def _resolved(rec: dict) -> bool | None:
    """eval records carry `resolved` (+ `error`); rollout Demos carry metadata.resolved."""
    if rec.get("error"):
        return None
    meta = rec.get("metadata") or {}
    v = rec["resolved"] if "resolved" in rec else meta.get("resolved")
    return None if v is None else bool(v)


def pass_counts(records: list[dict]) -> dict[str, tuple[int, int]]:
    """{instance: (resolved, graded)}; samples that errored (API/sandbox) are skipped."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        k, v = _key(r), _resolved(r)
        if k is None or v is None:
            continue
        out[k][0] += v
        out[k][1] += 1
    return {k: (s, n) for k, (s, n) in out.items()}


def teacher_status(records: list[dict]) -> dict[str, bool]:
    """Teacher solved the instance at least once (eval records or distill rollouts)."""
    return {k: s > 0 for k, (s, n) in pass_counts(records).items() if n}


def bucket_of(p: float, teacher: bool | None, low: float = 0.0, high: float = 1.0) -> str:
    if p >= high:
        return "mastered"
    if p <= low:
        return "unsolved" if teacher is False else "gap"
    return "learning"


def sft_weight(p: float, bucket: str, floor: float = 0.2) -> float:
    return 0.0 if bucket == "unsolved" else round(max(floor, 1.0 - p), 4)


def compute_weights(student: list[dict], teacher: list[dict] | None = None, low: float = 0.0, high: float = 1.0,
                    floor: float = 0.2, min_samples: int = 1) -> dict[str, Any]:
    counts = pass_counts(student)
    tstat = teacher_status(teacher or [])
    inst = {}
    for k, (s, n) in sorted(counts.items()):
        if n < min_samples:
            continue
        p = s / n
        b = bucket_of(p, tstat.get(k), low, high)
        inst[k] = {"resolved": s, "samples": n, "p": round(p, 4), "teacher_resolved": tstat.get(k),
                   "bucket": b, "weight": sft_weight(p, b, floor)}
    summary = {b: sum(v["bucket"] == b for v in inst.values()) for b in BUCKETS}
    summary["instances"] = len(inst)
    summary["mean_p"] = round(sum(v["p"] for v in inst.values()) / len(inst), 4) if inst else None
    return {"params": {"low": low, "high": high, "floor": floor, "min_samples": min_samples},
            "summary": summary, "instances": inst}


# ---- consumers ------------------------------------------------------------------------------------
def load_weights(path: str | Path | None) -> dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return data.get("instances", data)


def keep_for_teacher(key: str, weights: dict[str, dict], max_pass_rate: float) -> bool:
    """Instances with no measurement are kept (we don't know the student solves them)."""
    w = weights.get(key)
    return w is None or w["p"] <= max_pass_rate


def record_instance(rec: dict) -> str:
    meta = rec.get("meta") or rec.get("metadata") or {}
    return str(meta.get("parent_id") or meta.get("instance_id") or rec.get("instance_id")
               or str(rec.get("id", "")).split("#")[0])


def apply_instance_weights(records: list[dict], weights: dict[str, dict], unknown_weight: float = 1.0,
                           max_per_instance: float | None = 3.0) -> tuple[list[dict], dict]:
    """Scale each trajectory record's loss weight by its instance weight, drop zero-weight records,
    then cap the summed weight per instance. Aux-task records (source aux:*) are left unchanged."""
    out, dropped, scaled = [], 0, 0
    for r in records:
        if str(r.get("source", "")).startswith("aux:"):
            out.append(r)
            continue
        w = weights.get(record_instance(r))
        f = unknown_weight if w is None else float(w["weight"])
        if f <= 0:
            dropped += 1
            continue
        r = {**r, "weight": round(float(r.get("weight", 1.0)) * f, 4), "instance_weight": f}
        out.append(r)
    if max_per_instance:
        totals: dict[str, float] = defaultdict(float)
        for r in out:
            if "instance_weight" in r:
                totals[record_instance(r)] += r["weight"]
        for r in out:
            t = totals.get(record_instance(r), 0.0)
            if "instance_weight" in r and t > max_per_instance:
                r["weight"] = round(r["weight"] * max_per_instance / t, 4)
                scaled += 1
    return out, {"dropped_zero_weight": dropped, "capped_records": scaled, "kept": len(out)}


def select_student_successes(rollouts: list, per_instance: int = 1) -> list:
    """The student's own resolved episodes (rejection-sampled self-distillation), shortest first."""
    by: dict[str, list] = defaultdict(list)
    for d in rollouts:
        if d.metadata.get("resolved") is True and d.trajectory:
            by[str(d.metadata.get("parent_id") or d.instance_id)].append(d)
    return [d for k in sorted(by) for d in sorted(by[k], key=lambda d: len(d.trajectory))[:per_instance]]


# ---- CLI ------------------------------------------------------------------------------------------
def format_summary(w: dict) -> str:
    s, p = w["summary"], w["params"]
    lines = [f"{s['instances']} instances, mean student pass rate {s['mean_p']}"]
    what = {"gap": "teacher data / on-policy relabel", "learning": "RL + DPO", "mastered": f"replay at {p['floor']}",
            "unsolved": "skipped (no correct run)"}
    lines += [f"  {b:9} {s[b]:5}  -> {what[b]}" for b in BUCKETS]
    return "\n".join(lines)


def cmd_weights(a) -> int:
    student = [r for path in a.student for r in read_jsonl(path)]
    teacher = [r for path in (a.teacher or []) for r in read_jsonl(path)]
    w = compute_weights(student, teacher, a.low, a.high, a.floor, a.min_samples)
    if not w["instances"]:
        print("no graded student samples found", file=sys.stderr)
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(w, indent=2))
    if a.pass_rates_out:   # rl-rollouts --pass-rates: fractions of k
        Path(a.pass_rates_out).write_text(json.dumps({k: float(v["p"]) for k, v in w["instances"].items()}, indent=2))
    print(format_summary(w))
    print(f"wrote {a.out}" + (f" and {a.pass_rates_out}" if a.pass_rates_out else ""))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("difficulty-weights",
                              help="per-instance training weights from the student's pass rate (free)")
    p.add_argument("student", nargs="+", help="student eval JSONL (--samples k) or saved rollouts")
    p.add_argument("--teacher", nargs="+", default=None, help="teacher eval JSONL or distill rollouts (optional)")
    p.add_argument("--out", required=True, help="weights JSON")
    p.add_argument("--pass-rates-out", default=None, help="also write {instance: p} for rl-rollouts --pass-rates")
    p.add_argument("--low", type=float, default=0.0, help="p <= low counts as not solved by the student")
    p.add_argument("--high", type=float, default=1.0, help="p >= high counts as mastered")
    p.add_argument("--floor", type=float, default=0.2, help="minimum SFT weight for instances with a correct run")
    p.add_argument("--min-samples", type=int, default=1, help="ignore instances with fewer graded samples")
    p.set_defaults(fn=cmd_weights)
