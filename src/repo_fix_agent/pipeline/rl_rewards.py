"""Reward components for RL beyond pass/fail (used by rl.compute_reward / RepoFixEnv).

Shaped credit (hidden tests only: the grader applies the held-out test_patch last, so the policy
can't edit or see the tests it is scored on):
  resolved                       1.0
  not resolved                   partial_weight * (FAIL_TO_PASS passed / total)
  any                            - p2p_penalty * (PASS_TO_PASS broken / total)
A resolved episode always outranks an unresolved one (max unresolved credit is partial_weight +
repro_bonus < 1).

Verifiable repro bonus: the student's own `scratch_*.py` scripts are run twice, against the
untouched base commit and against its patched tree. One script that FAILS on the base (a real
failure, not an import / syntax / name error) and PASSES on the patch is evidence it reproduced
the bug and checked the fix: the habit that took baseline v3 from 83% to 88%. Scripts never ship
in the patch (env excludes scratch_*), so this bonus can't change what is graded.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SCRIPTS = 5
MAX_SCRIPT_BYTES = 20_000
# A failure caused by the script not running at all doesn't show the bug was reproduced. AttributeError
# and TypeError are left in: "the method doesn't handle X yet" is a legitimate failing repro.
TRIVIAL_FAILURE = re.compile(r"\b(ModuleNotFoundError|ImportError|SyntaxError|IndentationError|NameError|"
                             r"FileNotFoundError: \[Errno 2\] No such file or directory: '[^']*scratch_)")


def shaped_credit(resolved: bool, grade: dict | None, partial_weight: float = 0.5,
                  p2p_penalty: float = 0.5) -> dict[str, float]:
    if not grade or grade.get("error"):
        return {"partial_credit": 0.0, "p2p_penalty": 0.0}
    f2p, p2p = grade.get("fail_to_pass") or {}, grade.get("pass_to_pass") or {}
    partial = 0.0 if resolved or not f2p else partial_weight * sum(map(bool, f2p.values())) / len(f2p)
    broken = sum(not v for v in p2p.values())
    return {"partial_credit": round(partial, 4),
            "p2p_penalty": -round(p2p_penalty * broken / len(p2p), 4) if p2p else 0.0}


@dataclass
class ScriptRun:
    path: str
    base_rc: int | None = None
    patched_rc: int | None = None
    base_trivial: bool = False
    base_tail: str = ""

    @property
    def valid(self) -> bool:
        return self.base_rc not in (None, 0) and not self.base_trivial and self.patched_rc == 0


@dataclass
class ReproResult:
    scripts: list[ScriptRun] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return any(s.valid for s in self.scripts)

    def summary(self) -> dict[str, Any]:
        return {"scripts": len(self.scripts), "valid_repro": self.valid,
                "runs": [{"path": s.path, "base_rc": s.base_rc, "patched_rc": s.patched_rc,
                          "base_trivial": s.base_trivial} for s in self.scripts]}


def collect_scratch(env) -> dict[str, str]:
    """scratch_*.py files the agent left in its sandbox (top level and one directory down)."""
    r = env.run("ls -1 scratch_*.py */scratch_*.py 2>/dev/null | head -n 50", timeout=30)
    out: dict[str, str] = {}
    for p in [ln.strip() for ln in r.output.splitlines() if ln.strip().endswith(".py")][:MAX_SCRIPTS]:
        try:
            text = env.sandbox.read_file(p)
        except Exception:  # noqa: BLE001 - unreadable file: skip it
            continue
        if len(text.encode()) <= MAX_SCRIPT_BYTES:
            out[p] = text
    return out


def _run_script(env, path: str, timeout: int) -> tuple[int, str]:
    d, name = str(Path(path).parent), Path(path).name
    r = env.run(f"cd {shlex.quote(d)} && timeout {timeout} python {shlex.quote(name)}", timeout=timeout + 15)
    return r.returncode, r.output[-2000:]


def run_patched(env, scripts: dict[str, str], timeout: int = 30) -> dict[str, int]:
    return {p: _run_script(env, p, timeout)[0] for p in scripts}


def run_base(repo: str, base_commit: str | None, scripts: dict[str, str], backend: str = "auto",
             timeout: int = 30) -> dict[str, tuple[int, str]]:
    from ..env import RepoEnv

    with RepoEnv(repo, base_commit, backend=backend) as base:
        out = {}
        for p, text in scripts.items():
            base.sandbox.write_file(p, text)
            out[p] = _run_script(base, p, timeout)
        return out


def repro_check(patched_rc: dict[str, int], base: dict[str, tuple[int, str]]) -> ReproResult:
    runs = []
    for p, rc in patched_rc.items():
        brc, bout = base.get(p, (None, ""))
        runs.append(ScriptRun(p, brc, rc, bool(TRIVIAL_FAILURE.search(bout or "")), (bout or "")[-300:]))
    return ReproResult(runs)
