"""Baseline evaluation: run a model as the agent on demo instances and grade every patch.

    repo-fix eval data/demos/seed.jsonl --model student --workers 8 --out runs/baseline/student.jsonl

Writes one JSON line per instance (resolved, steps, error, seconds, difficulty) and a summary
(overall + by difficulty + tokens and estimated API cost) next to it. LLM calls go through the
response cache, so re-running the same evaluation is free; set REPOFIX_LLM_CACHE=off to resample.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .agent import run_agent
from .env import RepoEnv
from .grader import grade
from .pipeline.parallel import pmap
from .schema import Demo

# $ per million tokens: input, output, cache write (5-minute), cache read. None = unknown.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
}


def estimate_cost(model: str, usage: dict) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage.get("input", 0) * p[0] + usage.get("output", 0) * p[1] + usage.get("cache_write", 0) * p[2]
            + usage.get("cache_read", 0) * p[3]) / 1e6


def evaluate(demos: list[Demo], policy_factory: Callable[[], object], root: str | Path = ".", max_steps: int = 40,
             workers: int = 1, on_result: Callable[[dict], None] | None = None, samples: int = 1,
             source: str = "unknown", on_rollout: Callable[[Demo], None] | None = None,
             best_of: int = 1) -> list[dict]:
    """One record per (instance, sample). `on_rollout` receives each episode as a Demo with the
    trajectory, patch and metadata (source, parent_id, sample_idx, resolved): the format
    `prefs`, `verifier-train` and on-policy distillation consume."""
    def one(job: tuple[Demo, int]) -> dict:
        d, k = job
        t0 = time.time()
        rec = {"instance_id": d.instance_id, "sample": k, "difficulty": d.metadata.get("difficulty", "unknown"),
               "resolved": False, "steps": 0, "submitted": False, "error": None}
        try:
            repo = d.repo if "://" in d.repo else str(Path(root) / d.repo)
            if best_of > 1:
                return _best_of(d, k, rec, t0)
            with RepoEnv(repo, d.base_commit) as env:
                res = run_agent(policy_factory(), env, d.problem_statement, d.test_cmd, max_steps)
            g = grade(d, res.patch, root)
            if on_rollout:
                on_rollout(dataclasses.replace(d, trajectory=res.trajectory, patch=res.patch, metadata={
                    **{x: d.metadata[x] for x in ("difficulty", "bug_type", "repo_group") if x in d.metadata},
                    "source": source, "parent_id": d.instance_id, "sample_idx": k, "resolved": g.resolved,
                    "submitted": res.submitted, "num_steps": len(res.trajectory),
                    "grade": {"fail_to_pass": g.fail_to_pass, "pass_to_pass": g.pass_to_pass}}))
            rec.update(resolved=g.resolved, steps=len(res.trajectory), submitted=res.submitted,
                       patch_lines=sum(1 for ln in res.patch.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")),
                       f2p_passed=sum(g.fail_to_pass.values()), f2p_total=len(g.fail_to_pass),
                       p2p_broken=sum(not v for v in g.pass_to_pass.values()), grade_error=g.error)
        except Exception as e:  # noqa: BLE001 - one API/sandbox failure must not kill the run
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["seconds"] = round(time.time() - t0, 1)
        if on_result:
            on_result(rec)
        return rec

    def _best_of(d: Demo, k: int, rec: dict, t0: float) -> dict:
        from .pipeline.rerank import best_of_n_episode

        ranked, cands = best_of_n_episode(policy_factory, d, best_of, root, max_steps)
        chosen = ranked[0].patch if ranked and ranked[0].score >= 0 else ""
        graded: dict[str, bool] = {}
        for c in cands:   # oracle: did any candidate solve it? (selection loss = oracle - chosen)
            if c.patch.strip() and c.patch not in graded:
                graded[c.patch] = grade(d, c.patch, root).resolved
        rec.update(resolved=graded.get(chosen, False) if chosen else False, best_of=best_of,
                   oracle_resolved=any(graded.values()), distinct_patches=len(graded),
                   steps=sum(len(c.trajectory) for c in cands), submitted=bool(chosen),
                   chosen_score=round(ranked[0].score, 3) if ranked else None, seconds=round(time.time() - t0, 1))
        if on_result:
            on_result(rec)
        return rec

    return pmap(one, [(d, k) for d in demos for k in range(samples)], workers)


def summarize(model: str, records: list[dict], usage: dict | None = None) -> dict:
    """With several samples per instance: `accuracy` is mean pass@1 over all samples and
    `pass_at_k` the fraction of instances solved by at least one sample (the best-of-n ceiling)."""
    by = defaultdict(lambda: [0, 0])
    for r in records:
        by[r["difficulty"]][0] += r["resolved"]
        by[r["difficulty"]][1] += 1
    n, ok = len(records), sum(r["resolved"] for r in records)
    per_inst: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        per_inst[r["instance_id"]].append(bool(r["resolved"]))
    k = max((len(v) for v in per_inst.values()), default=1)
    errored = [r for r in records if r["error"]]
    scored = n - len(errored)
    out = {"model": model, "n": n, "resolved": ok, "accuracy": ok / n if n else 0.0,
           "accuracy_excluding_errors": ok / scored if scored else 0.0, "errors": len(errored),
           "by_difficulty": {k: {"resolved": v[0], "n": v[1], "accuracy": v[0] / v[1]}
                             for k, v in sorted(by.items(), key=lambda kv: ["easy", "medium", "hard"].index(kv[0])
                                                if kv[0] in ("easy", "medium", "hard") else 9)},
           "mean_steps": sum(r["steps"] for r in records) / n if n else 0.0,
           "submitted": sum(r["submitted"] for r in records),
           "error_kinds": dict(Counter((r["error"] or "").split(":")[0] for r in errored)),
           "samples_per_instance": k, "instances": len(per_inst)}
    if any("best_of" in r for r in records):
        out["best_of"] = max(r.get("best_of", 1) for r in records)
        out["oracle"] = sum(r.get("oracle_resolved", False) for r in records) / n if n else 0.0
    if k > 1:
        out["pass_at_k"] = sum(any(v) for v in per_inst.values()) / len(per_inst)
        out["solved_sometimes"] = sum(any(v) and not all(v) for v in per_inst.values())   # RL-informative
    if usage is not None:
        out["usage"] = dict(usage)
        out["est_cost_usd"] = estimate_cost(model, usage)
    return out


def format_summary(s: dict) -> str:
    head = (f"{s['model']}: mean pass@1 {s['accuracy']:.1%} over {s['samples_per_instance']} samples x "
            f"{s['instances']} instances, pass@{s['samples_per_instance']} {s['pass_at_k']:.1%}, "
            f"{s['solved_sometimes']} solved only sometimes" if s.get("pass_at_k") is not None else None)
    lines = [head or f"{s['model']}: {s['resolved']}/{s['n']} resolved = {s['accuracy']:.1%}"
             + (f"  ({s['errors']} errored; {s['accuracy_excluding_errors']:.1%} of the rest)" if s["errors"] else "")]
    if s.get("best_of"):
        lines.append(f"  best-of-{s['best_of']} (exec rerank): chosen {s['accuracy']:.1%}, oracle (any candidate) "
                     f"{s['oracle']:.1%}, selection loss {s['oracle'] - s['accuracy']:.1%}")
    lines += [f"  {k:7} {v['resolved']:3}/{v['n']:<3} {v['accuracy']:.0%}" for k, v in s["by_difficulty"].items()]
    lines.append(f"  mean steps {s['mean_steps']:.1f}, submitted {s['submitted']}/{s['n']}")
    if s.get("usage"):
        u = s["usage"]
        cost = f", est. ${s['est_cost_usd']:.2f}" if s.get("est_cost_usd") is not None else ""
        lines.append(f"  API (live calls only): {u.get('calls', 0)} calls, tokens in {u.get('input', 0):,} "
                     f"/ cache-write {u.get('cache_write', 0):,} / cache-read {u.get('cache_read', 0):,} "
                     f"/ out {u.get('output', 0):,}{cost}")
    return "\n".join(lines)
