"""Execution-based grading: a fix counts only if hidden tests flip and nothing regresses.

This is the single source of truth for "success rate" (student ~30%, teacher ~60%),
for rejection-sampling teacher data, and for the RL reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .env import RepoEnv
from .schema import Demo


@dataclass
class GradeResult:
    resolved: bool
    fail_to_pass: dict[str, bool] = field(default_factory=dict)
    pass_to_pass: dict[str, bool] = field(default_factory=dict)
    error: str | None = None


def _run_tests(env: RepoEnv, test_cmd: str, test_ids: list[str]) -> dict[str, bool]:
    # one process per test keeps results unambiguous
    return {tid: env.run(f"{test_cmd} {tid}", timeout=300).returncode == 0 for tid in test_ids}


def grade(demo: Demo, candidate_patch: str, dataset_root: str | Path = ".") -> GradeResult:
    repo = demo.repo if "://" in demo.repo else str(Path(dataset_root) / demo.repo)
    with RepoEnv(repo, demo.base_commit) as env:
        try:
            env.apply_patch(candidate_patch)
        except RuntimeError as e:
            return GradeResult(False, error=f"candidate patch failed to apply: {e}")
        env.apply_patch(demo.test_patch)  # hidden tests go on last so they can't be edited
        f2p = _run_tests(env, demo.test_cmd, demo.FAIL_TO_PASS)
        p2p = _run_tests(env, demo.test_cmd, demo.PASS_TO_PASS)
    return GradeResult(all(f2p.values()) and all(p2p.values()), f2p, p2p)


def validate_instance(demo: Demo, dataset_root: str | Path = ".") -> list[str]:
    """Dataset QA: gold patch must flip FAIL_TO_PASS and keep PASS_TO_PASS green."""
    problems = []
    before = grade(demo, "", dataset_root)
    if any(before.fail_to_pass.values()):
        problems.append(f"FAIL_TO_PASS already passing without fix: {before.fail_to_pass}")
    if not all(before.pass_to_pass.values()):
        problems.append(f"PASS_TO_PASS failing before fix: {before.pass_to_pass}")
    after = grade(demo, demo.patch, dataset_root)
    if not after.resolved:
        problems.append(f"gold patch does not resolve: {after}")
    return problems
