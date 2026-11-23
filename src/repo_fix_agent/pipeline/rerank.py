"""Execution-based reranking for best-of-n (no trained verifier, no extra model calls).

The student is prompted to write `scratch_*.py` reproduction scripts (agent.SPEC_CHECK). Given n
candidate episodes for one issue, we pool every candidate's scripts and run each candidate patch
against the visible test suite and against *all* pooled scripts (CodeT-style cross-checking: a
patch that satisfies other attempts' checks is more likely right than one that only satisfies
its own). Score = suite_weight * suite passes + check_weight * fraction of checks passing +
vote_weight * fraction of candidates with the same normalized patch. Empty patches rank last.

Used by `repo-fix best-of-n --scorer exec` and `repo-fix eval --best-of N`.
"""
from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..env import RepoEnv
from ..schema import Demo
from .verifier import normalize_patch, record_patch

MAX_CHECKS = 12


@dataclass
class Check:
    name: str
    content: str
    origin: int          # candidate index that wrote it


@dataclass
class Ranked:
    index: int
    record: Demo
    patch: str
    score: float
    suite_ok: bool | None
    checks_passed: int
    checks_total: int
    agreement: float
    detail: dict = field(default_factory=dict)


def extract_checks(rec: Demo, origin: int = 0) -> list[Check]:
    """Final content of every scratch_*.py the episode created (later edits via str_replace /
    create_file overwrite earlier versions)."""
    files: dict[str, str] = {}
    for st in rec.trajectory:
        a = st.action.args or {}
        path = str(a.get("path", ""))
        base = path.rsplit("/", 1)[-1]
        if not (base.startswith("scratch_") and base.endswith(".py")):
            continue
        if st.action.tool == "create_file" and isinstance(a.get("content"), str):
            files[path] = a["content"]
        elif st.action.tool == "str_replace" and path in files and a.get("old_str") in files[path]:
            files[path] = files[path].replace(a["old_str"], a.get("new_str", ""), 1)
    return [Check(p, c, origin) for p, c in files.items() if c.strip()]


def pool_checks(candidates: Sequence[Demo], limit: int = MAX_CHECKS) -> list[Check]:
    seen, out = set(), []
    for i, c in enumerate(candidates):
        for chk in extract_checks(c, i):
            h = hashlib.sha1(chk.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(chk)
    return out[:limit]


def rerank(candidates: Sequence[Demo], repo: str | Path, base_commit: str | None, test_cmd: str | None,
           backend: str = "auto", suite_weight: float = 0.4, check_weight: float = 0.4, vote_weight: float = 0.2,
           timeout: int = 120, env_factory: Callable[..., RepoEnv] | None = None) -> list[Ranked]:
    env_factory = env_factory or (lambda: RepoEnv(str(repo), base_commit, backend=backend))
    checks = pool_checks(candidates)
    keys = [normalize_patch(record_patch(c)) for c in candidates]
    counts: dict[str, int] = {}
    for k in keys:
        if k:
            counts[k] = counts.get(k, 0) + 1
    cache: dict[str, tuple[bool | None, list[bool]]] = {}   # identical patches are executed once
    ranked = []
    for i, c in enumerate(candidates):
        patch = record_patch(c)
        if not keys[i]:
            ranked.append(Ranked(i, c, patch, -1.0, None, 0, len(checks), 0.0, {"empty_patch": True}))
            continue
        if keys[i] not in cache:
            cache[keys[i]] = _execute(env_factory, patch, checks, test_cmd, timeout)
        suite_ok, results = cache[keys[i]]
        frac = sum(results) / len(results) if results else 0.0
        agree = counts[keys[i]] / len(candidates)
        score = suite_weight * (1.0 if suite_ok else 0.0) + check_weight * frac + vote_weight * agree
        ranked.append(Ranked(i, c, patch, score, suite_ok, sum(results), len(results), agree,
                             {"checks": {chk.name + f"@{chk.origin}": ok for chk, ok in zip(checks, results)}}))
    ranked.sort(key=lambda r: (-round(r.score, 9), r.index))
    return ranked


def _execute(env_factory, patch: str, checks: list[Check], test_cmd: str | None, timeout: int):
    with env_factory() as env:
        try:
            env.apply_patch(patch)
        except RuntimeError:
            return False, [False] * len(checks)
        suite_ok = env.run(test_cmd, timeout=timeout).returncode == 0 if test_cmd else None
        results = []
        for k, chk in enumerate(checks):
            name = f"scratch_rerank_{k}.py"     # scratch_* is excluded from patches
            env.sandbox.write_file(name, chk.content)
            results.append(env.run(f"python {shlex.quote(name)}", timeout=min(timeout, 60)).returncode == 0)
    return suite_ok, results


def best_of_n_episode(policy_factory: Callable[[], object], demo: Demo, n: int, root: str | Path = ".",
                      max_steps: int = 40, workers: int = 1, backend: str = "auto") -> tuple[list[Ranked], list[Demo]]:
    """Sample n episodes for `demo` (concurrently with workers > 1) and rerank them."""
    from ..agent import run_agent
    from .parallel import pmap

    repo = demo.repo if "://" in demo.repo else str(Path(root) / demo.repo)

    def one(k: int) -> Demo:
        with RepoEnv(repo, demo.base_commit, backend=backend) as env:
            res = run_agent(policy_factory(), env, demo.problem_statement, demo.test_cmd, max_steps)
        return Demo(instance_id=f"{demo.instance_id}#s{k}", repo=demo.repo, base_commit=demo.base_commit,
                    problem_statement=demo.problem_statement, test_cmd=demo.test_cmd, test_patch="",
                    FAIL_TO_PASS=[], PASS_TO_PASS=[], patch=res.patch, trajectory=res.trajectory,
                    metadata={"sample_idx": k, "submitted": res.submitted, "parent_id": demo.instance_id})

    cands = pmap(one, range(n), workers)
    return rerank(cands, repo, demo.base_commit, demo.test_cmd, backend), cands
