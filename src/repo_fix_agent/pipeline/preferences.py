"""§3e preference data for DPO / KTO from resolved vs unresolved rollouts.

* Trajectory pairs: same parent instance, resolved (chosen) vs unresolved (rejected); student-student
  (on-policy) pairs are used when an instance has them, otherwise any source.
* Patch pairs: gold patch (chosen) vs failed rollout patch (rejected) in a single-turn
  "write the fix given the issue and the relevant code" format. The human patch is never spliced
  into a student trajectory.
* Step pairs: at the first action divergence between a resolved and an unresolved rollout of the same
  instance, chosen = resolved next action, rejected = unresolved next action, shared prefix as context.
* KTO: every rollout as an unpaired desirable / undesirable example.
* Length-bias guard: `length_stats` and `balance_lengths`.

Pair records use the TRL conversational format: {"prompt": [...], "chosen": [...], "rejected": [...]}.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

from .. import sft
from ..schema import Demo, load_demos
from .aux_tasks import RepoReader, relevant_source
from .sft_data import check_excluded, guard_frozen, instance_key, load_exclude_ids, patches_equivalent, write_jsonl

PATCH_SYSTEM = """You fix bugs. Given an issue and the relevant source code, write the fix as a unified diff
(`diff --git` format) against the repository. Output the diff only."""


def _len(msgs: list[dict]) -> int:
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in msgs)


def _is_student(d: Demo) -> bool:
    return str(d.metadata.get("source", "")).startswith("student:")


def _group(rollouts: list[Demo]) -> dict[str, tuple[list[Demo], list[Demo]]]:
    groups: dict[str, tuple[list[Demo], list[Demo]]] = {}
    for r in rollouts:
        res = r.metadata.get("resolved")
        if res is None:
            continue
        g = groups.setdefault(instance_key(r), ([], []))
        (g[0] if res else g[1]).append(r)
    return groups


def _pair_record(pid: str, kind: str, prompt, chosen, rejected, meta: dict, tools=None) -> dict:
    rec = {"id": pid, "type": kind, "prompt": prompt, "chosen": chosen, "rejected": rejected,
           "meta": {**meta, "chosen_len": _len(chosen), "rejected_len": _len(rejected)}}
    if tools is not None:
        rec["tools"] = tools
    return rec


def _candidate_pairs(res: list[Demo], unres: list[Demo]) -> list[tuple[Demo, Demo]]:
    """All resolved x unresolved pairs; restricted to student-student when any exist."""
    pairs = list(product(res, unres))
    on_policy = [(c, r) for c, r in pairs if _is_student(c) and _is_student(r)]
    return on_policy or pairs


# ---- trajectory-level -------------------------------------------------------------------------
def trajectory_pairs(rollouts: list[Demo], max_per_instance: int = 4) -> list[dict]:
    out = []
    for key, (res, unres) in sorted(_group(rollouts).items()):
        used: set[int] = set()
        n = 0
        for c, r in _candidate_pairs(res, unres):
            if n >= max_per_instance:
                break
            if id(c) in used or id(r) in used:  # each rollout used at most once per instance
                continue
            used |= {id(c), id(r)}
            cm, rm = sft.render_messages(c), sft.render_messages(r)
            out.append(_pair_record(f"{key}:traj{n}", "trajectory", cm[:2], cm[2:], rm[2:], {
                "instance_id": key, "chosen_source": c.metadata.get("source"),
                "rejected_source": r.metadata.get("source"),
                "on_policy": _is_student(c) and _is_student(r)}, sft.tools_as_functions()))
            n += 1
    return out


# ---- patch-level ------------------------------------------------------------------------------
def patch_prompt(demo: Demo, root: str | Path = ".") -> list[dict]:
    ctx = ""
    try:
        with RepoReader(demo, root) as reader:
            ctx = relevant_source(demo, reader)
    except (FileNotFoundError, RuntimeError):
        pass
    user = f"<issue>\n{demo.problem_statement.strip()}\n</issue>"
    if ctx:
        user += f"\n\nRelevant code:\n```\n{ctx}\n```"
    return [{"role": "system", "content": PATCH_SYSTEM}, {"role": "user", "content": user}]


def patch_pairs(demos: list[Demo], rollouts: list[Demo], root: str | Path = ".",
                max_per_instance: int = 4) -> list[dict]:
    groups = _group(rollouts)
    out = []
    for d in demos:
        failed = groups.get(d.instance_id, ([], []))[1]
        seen: list[str] = []
        rejected = []
        for r in sorted(failed, key=lambda x: not _is_student(x)):  # prefer on-policy negatives
            if not r.patch.strip() or patches_equivalent(r.patch, d.patch):
                continue
            if any(patches_equivalent(r.patch, s) for s in seen):
                continue
            seen.append(r.patch)
            rejected.append(r)
        if not rejected:
            continue
        prompt = patch_prompt(d, root)
        for k, r in enumerate(rejected[:max_per_instance]):
            out.append(_pair_record(f"{d.instance_id}:patch{k}", "patch", prompt,
                                    [{"role": "assistant", "content": d.patch}],
                                    [{"role": "assistant", "content": r.patch}],
                                    {"instance_id": d.instance_id, "chosen_source": "gold",
                                     "rejected_source": r.metadata.get("source")}))
    return out


# ---- step-level -------------------------------------------------------------------------------
def first_divergence(a: Demo, b: Demo) -> int:
    n = 0
    for sa, sb in zip(a.trajectory, b.trajectory):
        if sa.action != sb.action:
            break
        n += 1
    return n


def step_pairs(rollouts: list[Demo], max_per_instance: int = 4) -> list[dict]:
    out = []
    for key, (res, unres) in sorted(_group(rollouts).items()):
        seen: set[tuple] = set()
        n = 0
        for c, r in _candidate_pairs(res, unres):
            if n >= max_per_instance:
                break
            d = first_divergence(c, r)
            if d >= len(c.trajectory) or d >= len(r.trajectory):
                continue  # one is a prefix of the other: no rejected/chosen next action
            sig = (d, json.dumps(c.trajectory[d].action.__dict__, sort_keys=True),
                   json.dumps(r.trajectory[d].action.__dict__, sort_keys=True))
            if sig in seen:
                continue
            seen.add(sig)
            prefix_demo = Demo(**{**c.__dict__, "trajectory": c.trajectory[:d]})
            prompt = sft.render_messages(prefix_demo)
            cm = sft.render_messages(Demo(**{**c.__dict__, "trajectory": c.trajectory[:d + 1]}))[-2]
            rm = sft.render_messages(Demo(**{**r.__dict__, "trajectory": r.trajectory[:d + 1]}))[-2]
            out.append(_pair_record(f"{key}:step{n}", "step", prompt, [cm], [rm], {
                "instance_id": key, "prefix_len": d, "chosen_source": c.metadata.get("source"),
                "rejected_source": r.metadata.get("source")}, sft.tools_as_functions()))
            n += 1
    return out


# ---- KTO --------------------------------------------------------------------------------------
def kto_examples(rollouts: list[Demo]) -> list[dict]:
    out = []
    counts: dict[str, int] = {}
    for r in rollouts:
        if r.metadata.get("resolved") is None or not r.trajectory:
            continue
        key = instance_key(r)
        counts[key] = counts.get(key, 0) + 1
        msgs = sft.render_messages(r)
        out.append({"id": f"{key}:kto{counts[key] - 1}", "prompt": msgs[:2], "completion": msgs[2:],
                    "label": bool(r.metadata["resolved"]), "tools": sft.tools_as_functions(),
                    "meta": {"instance_id": key, "source": r.metadata.get("source"),
                             "length": _len(msgs[2:])}})
    return out


# ---- length-bias guard ------------------------------------------------------------------------
def length_stats(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0, "mean_chosen": 0.0, "mean_rejected": 0.0, "frac_chosen_shorter": 0.0}
    c = [p["meta"]["chosen_len"] for p in pairs]
    r = [p["meta"]["rejected_len"] for p in pairs]
    return {"n": len(pairs), "mean_chosen": sum(c) / len(c), "mean_rejected": sum(r) / len(r),
            "frac_chosen_shorter": sum(a < b for a, b in zip(c, r)) / len(pairs)}


def balance_lengths(pairs: list[dict]) -> list[dict]:
    """Drop pairs until chosen-shorter and chosen-longer pairs are equally common, removing the most
    extreme length ratios from the majority side first, so length alone doesn't predict the label."""
    def lr(p):
        return math.log((p["meta"]["chosen_len"] + 1) / (p["meta"]["rejected_len"] + 1))

    shorter = sorted((p for p in pairs if lr(p) < 0), key=lr)            # most negative first
    longer = sorted((p for p in pairs if lr(p) > 0), key=lr, reverse=True)
    drop: set[int] = set()
    if len(shorter) > len(longer):
        drop = {id(p) for p in shorter[: len(shorter) - len(longer)]}
    elif len(longer) > len(shorter):
        drop = {id(p) for p in longer[: len(longer) - len(shorter)]}
    return [p for p in pairs if id(p) not in drop]


# ---- driver -----------------------------------------------------------------------------------
def build_preferences(rollouts: list[Demo], demos: list[Demo] | None = None, root: str | Path = ".",
                      exclude_ids: set[str] | None = None, balance: bool = False,
                      max_per_instance: int = 4) -> dict:
    demos = demos or []
    check_excluded(rollouts, exclude_ids, "rollouts")
    check_excluded(demos, exclude_ids, "demos")
    traj = trajectory_pairs(rollouts, max_per_instance)
    patch = patch_pairs(demos, rollouts, root, max_per_instance)
    steps = step_pairs(rollouts, max_per_instance)
    stats = {"trajectory": length_stats(traj), "patch": length_stats(patch), "step": length_stats(steps)}
    if balance:
        traj, patch, steps = balance_lengths(traj), balance_lengths(patch), balance_lengths(steps)
        stats["balanced"] = {"trajectory": length_stats(traj), "patch": length_stats(patch),
                             "step": length_stats(steps)}
    kto = kto_examples(rollouts)
    stats["kto"] = {"n": len(kto), "desirable": sum(e["label"] for e in kto)}
    return {"dpo": traj + patch, "steps": steps, "kto": kto, "stats": stats}


def cmd_prefs(a) -> int:
    exclude = load_exclude_ids(a.exclude_ids)
    rollouts = [r for path in a.rollouts for r in load_demos(path)]
    demos = load_demos(a.demos) if a.demos else []
    out = build_preferences(rollouts, demos, a.root, exclude, a.balance_lengths)
    write_jsonl(out["dpo"], a.out_dpo)
    if a.out_kto:
        write_jsonl(out["kto"], a.out_kto)
    if a.out_steps:
        write_jsonl(out["steps"], a.out_steps)
    print(json.dumps(out["stats"]))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("prefs", help="§3e: DPO/KTO preference data from rollouts")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--demos", default=None, help="gold demos for patch-level pairs")
    p.add_argument("--out-dpo", default="runs/prefs/dpo.jsonl")
    p.add_argument("--out-kto", default=None)
    p.add_argument("--out-steps", default=None)
    p.add_argument("--balance-lengths", action="store_true")
    p.add_argument("--root", default=".")
    p.add_argument("--exclude-ids", default=None)
    p.set_defaults(fn=guard_frozen(cmd_prefs))
