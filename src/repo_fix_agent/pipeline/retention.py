"""Guarding against catastrophic forgetting: retention data, cross-round replay, and a forgetting gate.

Fine-tuning a small student on narrow bug-fixing data can erase what it already did well: other
instances it solved, its tool-call discipline, and general coding/chat ability. Five defences,
each measurable:

  1. LoRA only (scripts/train_lora.py): the base weights never change, and every reference /
     KL term below is the base with the adapter switched off.
  2. Retention KL (train_lora.py --retain-data --retain-beta): interleaved with the task loss, the
     adapter is penalised by KL(p_adapter || p_base) over the assistant tokens of retention
     records, so the fine-tuned model keeps the base's predictions where the task data says
     nothing. Retention records come from `retain-set`:
       * the student's own resolved episodes from before training (free; from rollouts or a run)
       * general prompts answered by the base student (bundled data/retention/general_prompts.jsonl;
         a few cheap Fireworks calls, dry run unless --yes)
  3. Replay across rounds (sft-mix --replay prev/sft.jsonl --replay-frac 0.2): a sample of the
     previous round's training records, so round 2 doesn't overwrite round 1.
  4. Forgetting gate (`forgetting-check --before <run> --after <run>`): instances the old model
     solved that the new one fails, tool-error rate per step, accuracy change; non-zero exit when
     a threshold is crossed. `student-train` runs it after the post-training eval.
  5. Weight interpolation (scripts/interpolate_lora.py --alpha 0.5): scale the adapter toward the
     base (WiSE-FT) when a fine-tune gains on the task but fails the gate.

  repo-fix retain-set --rollouts R.jsonl --out retain.jsonl          # free
  repo-fix retain-set --from-run "student baseline v3 ..." --out retain.jsonl
  repo-fix retain-set --general --out retain_general.jsonl [--yes]   # base student answers general prompts
  repo-fix forgetting-check --before "r1 dev before" --after "r1 dev after"
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

GENERAL_PROMPTS = Path(__file__).resolve().parents[3] / "data" / "retention" / "general_prompts.jsonl"


# ---- retention records ---------------------------------------------------------------------------
def episodes_to_retain(episodes: list[dict], cfg=None, resolved_only: bool = True) -> list[dict]:
    """Student episodes (rollout Demo dicts) -> chat records under the student's prompt. These are
    the behaviours to keep: what the student already did right, in its own words."""
    from ..schema import Demo
    from .sft_data import instance_key, render_record

    out = []
    for e in episodes:
        d = e if isinstance(e, Demo) else Demo.from_dict(e)
        if not d.trajectory or (resolved_only and not d.metadata.get("resolved")):
            continue
        r = render_record(d, cfg, "retain:episode", f"{instance_key(d)}#retain{d.metadata.get('sample_idx', 0)}")
        r["meta"] = {"parent_id": instance_key(d), "retain": True}
        out.append(r)
    return out


def load_general_prompts(path: str | Path = GENERAL_PROMPTS) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def general_records(prompts: list[dict], answer) -> list[dict]:
    """`answer(messages) -> str` is the BASE student (before training): its own answers are what
    the KL term preserves, so they must come from the model being fine-tuned, not a teacher."""
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        if p.get("system"):
            msgs.insert(0, {"role": "system", "content": p["system"]})
        reply = answer(msgs)
        if not reply.strip():
            continue
        out.append({"id": f"retain:{p['id']}", "messages": msgs + [{"role": "assistant", "content": reply}],
                    "tools": [], "loss_mask": [False] * len(msgs) + [True], "weight": 1.0,
                    "source": "retain:general", "meta": {"retain": True, "category": p.get("category")}})
    return out


def student_answerer(cfg, max_tokens: int = 1024):
    from openai import OpenAI

    from ..llm_cache import cached_call

    client = OpenAI(base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env) or "EMPTY")

    def answer(messages: list[dict]) -> str:
        req = {"model": cfg.model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        resp = cached_call(f"chat:{cfg.base_url}", req, lambda: client.chat.completions.create(**req))
        return resp.choices[0].message.content or ""
    return answer


# ---- replay across rounds -------------------------------------------------------------------------
def replay_sample(prev: list[dict], n_current: int, frac: float, weight: float = 1.0, seed: int = 0) -> list[dict]:
    """`frac` of the final mix comes from earlier rounds: n = frac/(1-frac) * n_current records."""
    if not prev or frac <= 0:
        return []
    n = min(len(prev), round(n_current * frac / (1 - frac)) if frac < 1 else len(prev))
    rng = random.Random(seed)
    out = []
    for r in rng.sample(prev, n):
        r = {**r, "weight": round(float(r.get("weight", 1.0)) * weight, 4), "source": f"replay:{r.get('source', '?')}"}
        r["id"] = f"replay:{r.get('id')}"
        out.append(r)
    return out


# ---- forgetting gate ------------------------------------------------------------------------------
def tool_error_rate(episodes: list[dict]) -> float | None:
    steps = [s for e in episodes for s in e.get("trajectory") or []]
    if not steps:
        return None
    return sum(1 for s in steps if str(s.get("observation", "")).startswith("Error:")) / len(steps)


def forgetting_report(before: list[dict], after: list[dict], eps_before: list[dict] | None = None,
                      eps_after: list[dict] | None = None) -> dict[str, Any]:
    from ..rundb import per_instance

    pb, pa = per_instance(before), per_instance(after)
    common = sorted(set(pb) & set(pa))
    rate = lambda v: sum(v) / len(v)  # noqa: E731
    solved_before = [i for i in common if rate(pb[i]) >= 0.5]
    forgotten = [i for i in solved_before if rate(pa[i]) < 0.5]
    learned = [i for i in common if rate(pb[i]) < 0.5 <= rate(pa[i])]
    mean = lambda p: sum(rate(p[i]) for i in common) / len(common) if common else None  # noqa: E731
    return {"common": len(common), "before": mean(pb), "after": mean(pa),
            "delta": (mean(pa) - mean(pb)) if common else None,
            "solved_before": len(solved_before), "forgotten": forgotten, "learned": learned,
            "forgotten_rate": len(forgotten) / len(solved_before) if solved_before else 0.0,
            "tool_error_rate_before": tool_error_rate(eps_before or []),
            "tool_error_rate_after": tool_error_rate(eps_after or [])}


def gate(rep: dict, max_forgotten_rate: float = 0.05, min_delta: float = 0.0,
         max_tool_error_increase: float = 0.02) -> list[str]:
    """Reasons the new model fails the gate (empty = passes)."""
    why = []
    if not rep["common"]:
        return ["the runs share no instances"]
    if rep["forgotten_rate"] > max_forgotten_rate:
        why.append(f"forgot {len(rep['forgotten'])}/{rep['solved_before']} previously solved instances "
                   f"({rep['forgotten_rate']:.1%} > {max_forgotten_rate:.0%})")
    if rep["delta"] is not None and rep["delta"] < min_delta:
        why.append(f"accuracy changed {rep['delta'] * 100:+.1f} pts (< {min_delta * 100:+.1f})")
    tb, ta = rep["tool_error_rate_before"], rep["tool_error_rate_after"]
    if tb is not None and ta is not None and ta - tb > max_tool_error_increase:
        why.append(f"tool-error rate per step rose {tb:.1%} -> {ta:.1%}")
    return why


# ---- CLI ------------------------------------------------------------------------------------------
def cmd_retain_set(a) -> int:
    from ..policies import StudentServingConfig
    from ..rundb import RunDB
    from ..schema import load_demos
    from .sft_data import check_excluded, load_exclude_ids, write_jsonl

    cfg = StudentServingConfig.load(a.config) if Path(a.config).exists() else StudentServingConfig()
    records: list[dict] = []
    if a.rollouts:
        records += episodes_to_retain([d.to_dict() for p in a.rollouts for d in load_demos(p)], cfg, not a.all)
    if a.from_run:
        db = RunDB(a.db)
        records += episodes_to_retain(db.episodes(a.from_run), cfg, not a.all)
        db.close()
    check_excluded(records, load_exclude_ids(a.exclude_ids), "retention records")
    if a.general:
        prompts = load_general_prompts(a.prompts)
        if not a.yes:
            print(f"general prompts: {len(prompts)} calls to the base student {cfg.model_name} "
                  "(dry run; re-run with --yes)", file=sys.stderr)
        else:
            records += general_records(prompts, student_answerer(cfg))
    if a.max_records and len(records) > a.max_records:
        records = random.Random(0).sample(records, a.max_records)
    if not records:
        print("no retention records (pass --rollouts / --from-run, or --general --yes)", file=sys.stderr)
        return 1
    n = write_jsonl(records, a.out)
    by = {}
    for r in records:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"wrote {n} retention records to {a.out} {by}")
    return 0


def cmd_forgetting(a) -> int:
    from ..rundb import RunDB

    db = RunDB(a.db)
    try:
        rb, ra = db.run(a.before), db.run(a.after)
        rep = forgetting_report(db.results(rb["run_id"]), db.results(ra["run_id"]),
                                db.episodes(rb["run_id"]), db.episodes(ra["run_id"]))
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    finally:
        db.close()
    why = gate(rep, a.max_forgotten_rate, a.min_delta, a.max_tool_error_increase)
    if a.json:
        print(json.dumps({**rep, "pass": not why, "reasons": why}, indent=2))
    else:
        print(f"before: {rb['label'] or rb['run_id']}\nafter:  {ra['label'] or ra['run_id']}")
        if rep["common"]:
            print(f"{rep['common']} instances: {rep['before']:.1%} -> {rep['after']:.1%} ({rep['delta'] * 100:+.1f} pts)")
            print(f"forgotten {len(rep['forgotten'])}/{rep['solved_before']} ({rep['forgotten_rate']:.1%}), "
                  f"newly solved {len(rep['learned'])}")
            if rep["forgotten"]:
                print("  forgotten: " + ", ".join(rep["forgotten"]))
            tb, ta = rep["tool_error_rate_before"], rep["tool_error_rate_after"]
            if tb is not None and ta is not None:
                print(f"tool errors per step: {tb:.1%} -> {ta:.1%}")
        print("PASS" if not why else "FAIL:\n  - " + "\n  - ".join(why))
    return 0 if not why else 1


def register(subparsers) -> None:
    r = subparsers.add_parser("retain-set", help="retention records for the anti-forgetting KL term")
    r.add_argument("--rollouts", nargs="+", default=None, help="student episodes from before training")
    r.add_argument("--from-run", default=None, help="take episodes from a run in the run database")
    r.add_argument("--db", default=None)
    r.add_argument("--all", action="store_true", help="keep unresolved episodes too (default: resolved only)")
    r.add_argument("--general", action="store_true", help="add general prompts answered by the base student")
    r.add_argument("--prompts", default=str(GENERAL_PROMPTS))
    r.add_argument("--yes", action="store_true", help="make the (paid) base-student calls for --general")
    r.add_argument("--max-records", type=int, default=None)
    r.add_argument("--exclude-ids", default=None)
    r.add_argument("--config", default="configs/models.yaml")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_retain_set)

    f = subparsers.add_parser("forgetting-check", help="did the new model forget what the old one solved? (gate)")
    f.add_argument("--before", required=True, help="run of the model before training (id / prefix / label)")
    f.add_argument("--after", required=True, help="run of the trained model on the same instances")
    f.add_argument("--db", default=None)
    f.add_argument("--max-forgotten-rate", type=float, default=0.05,
                   help="fail if more than this share of previously solved instances now fail")
    f.add_argument("--min-delta", type=float, default=0.0, help="fail if accuracy changes by less than this")
    f.add_argument("--max-tool-error-increase", type=float, default=0.02,
                   help="fail if tool errors per step rise by more than this")
    f.add_argument("--json", action="store_true")
    f.set_defaults(fn=cmd_forgetting)
