"""RepoEnv: one agent episode = a sandboxed repo copy + the shared tool layer.

  * sandbox.py - where commands run (LocalSandbox for dev, DockerSandbox for production:
                 no network, memory/pid/cpu limits). Picked by `backend`.
  * tools.py   - what the agent can do (validated actions -> observations).

The agent never touches the original checkout; its answer is `git diff` against the sandbox's
baseline commit, with test files excluded so it can't game the hidden tests.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from .sandbox import ExecResult, SandboxLimits, make_sandbox
from .schema import Action
from .tools import ToolState, execute_tool

_PATCH_FILE = ".git/repofix-incoming.patch"
_DIFF_FILE = ".git/repofix-outgoing.diff"
_TEST_EXCLUDES = ("':(exclude)tests/**' ':(exclude)**/test_*.py' ':(exclude)**/*_test.py' "
                  "':(exclude)scratch_*' ':(exclude)**/scratch_*'")   # agent repro scripts never ship


_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_VOLATILE = (
    (re.compile(rf"\b{_MON}\s+\d{{1,2}}\s+(?:\d{{1,2}}:\d{{2}}|\d{{4}})\b"), "Jan  1 00:00"),  # ls -l dates
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "2000-01-01 00:00:00"),  # timestamps
    (re.compile(r"\bin \d+(?:\.\d+)?s\b"), "in 0.00s"),                 # pytest / unittest run time
    (re.compile(r"\b\d+\.\d+s(?= (?:call|setup|teardown)\b)"), "0.00s"),  # pytest --durations
    (re.compile(r"\b0x[0-9a-fA-F]{8,}\b"), "0x0"),                          # object addresses in reprs
)


def normalize_observation(text: str) -> str:
    """Mask run-to-run noise in tool output (file times, timestamps, durations, memory addresses).
    None of it matters for fixing a bug, but it changes the prompt, which defeats the LLM cache:
    baseline v3 vs v5 diverged in 69 of 100 episodes, mostly at an `ls -la` date or a pytest
    duration, so scaffold comparisons weren't the paired comparisons they were meant to be."""
    for rx, rep in _VOLATILE:
        text = rx.sub(rep, text)
    return text


class RepoEnv:
    def __init__(self, repo: str | Path, base_commit: str | None = None, backend: str = "auto",
                 limits: SandboxLimits | None = None, image: str | None = None):
        self.sandbox = make_sandbox(str(repo), base_commit, backend=backend, limits=limits, image=image)
        self.state = ToolState()

    @property
    def done(self) -> bool:
        return self.state.submitted

    @property
    def baseline(self) -> str:
        return self.sandbox.baseline

    def execute(self, action: Action) -> str:
        return normalize_observation(execute_tool(self.sandbox, self.state, action.tool, action.args).text)

    def run(self, command: str, timeout: int = 120) -> ExecResult:
        """Run a command in the sandbox outside the agent's action space (grading, setup)."""
        return self.sandbox.exec(command, timeout=timeout)

    def diff(self, exclude_tests: bool = True) -> str:
        # Written to a file and read back: exec output is truncated, patches must not be.
        spec = f". {_TEST_EXCLUDES}" if exclude_tests else "."
        r = self.run(f"git add -A && git diff --cached {self.baseline} -- {spec} > {_DIFF_FILE}")
        if r.returncode != 0:
            raise RuntimeError(f"git diff failed: {r.output}")
        return self.sandbox.read_file(_DIFF_FILE)

    def apply_patch(self, patch: str) -> None:
        if not patch.strip():
            return
        self.sandbox.write_file(_PATCH_FILE, patch)
        r = self.run(f"git apply --whitespace=nowarn {shlex.quote(_PATCH_FILE)}; rc=$?; rm -f {_PATCH_FILE}; exit $rc")
        if r.returncode != 0:
            raise RuntimeError(f"git apply failed: {r.output}")

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> "RepoEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
