"""The agent loop: plug any Policy onto any repo."""
from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Callable

from .env import RepoEnv
from .policies import Policy
from .schema import Step

SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in the repository you are given.

Work like a careful engineer:
1. Locate the relevant code (grep, view files). Read before you edit.
2. Reproduce the bug with a short command or script when practical.
3. Make the smallest correct change to source code. Do not modify existing tests to make them pass.
4. Re-run the reproduction and the existing test suite to verify the fix and check for regressions.
5. Call `submit` when done.

Before each tool call, write one or two sentences on what you are checking or changing and why."""


SPEC_CHECK = """

Before you edit, list every behaviour the issue implies: the case it shows, plus the edge cases
around it (empty or boundary inputs, other call paths that share the buggy code, related
methods). Write a small reproduction script named `scratch_<something>.py` that checks each of
them; it fails before your fix and passes after. Scratch files are never part of your patch.
Passing existing tests is not enough on its own: they often miss the edge cases."""

SUBMIT_NUDGE = (
    "Not submitted yet. You haven't run a reproduction script since your last edit, and passing the "
    "existing tests is not enough: they often miss the edge cases. Before submitting: list the "
    "behaviours the issue implies (the case it shows, boundary and empty inputs, other call paths that "
    "share the code you changed), write a scratch_*.py script that checks each of them, run it, and run "
    "the test suite. If everything holds, call `submit` again.")


@dataclass
class AgentOptions:
    """Inference scaffold options. The student's come from configs/models.yaml (student.serving)
    and are reused for SFT rendering, RL and distillation so training matches inference."""
    spec_check: bool = False     # extend the system prompt with the behaviour/edge-case checklist
    submit_gate: bool = False    # bounce the first `submit` that follows an unverified edit
    step_notice: int = 0         # append "[N steps left]" to observations once N <= this (0 = off)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT + (SPEC_CHECK if self.spec_check else "")


def steps_left_notice(left: int, threshold: int) -> str:
    """Suffix for an observation when the step budget is nearly spent (`left` = actions remaining).
    Baseline v3 lost 3 instances to the step limit; the student had no way to see it coming."""
    if not threshold or left > threshold or left < 0:
        return ""
    if left == 0:
        return ""
    if left == 1:
        return "\n\n[1 step left: this is your last action. Call `submit` now if your fix is in place.]"
    return f"\n\n[{left} steps left. Leave time to verify with your scratch script and submit.]"


def student_options(path: str = "configs/models.yaml") -> AgentOptions:
    from .policies import StudentServingConfig
    try:
        return StudentServingConfig.load(path).agent_options()
    except FileNotFoundError:
        return AgentOptions()


_EDIT_TOOLS = {"str_replace", "create_file", "insert", "undo_edit"}
_SCRATCH_RUN = re.compile(r"(python[\d.]*|pytest)\b[^|;&]*\bscratch_\w*\.py")


def _is_check(tool: str, args: dict) -> bool:
    """A step that runs a scratch_*.py reproduction script. Running the existing test suite does
    not count: the student's failures pass the visible tests and miss hidden edge cases
    (baseline v2: the gate never fired because every episode ran the suite after editing)."""
    return tool == "bash" and bool(_SCRATCH_RUN.search(str(args.get("command", ""))))


class SubmitGate:
    """Bounces at most one `submit` per episode when no scratch_*.py repro script has run since
    the last source edit."""

    def __init__(self, enabled: bool):
        self.enabled, self.used = enabled, False
        self.edited, self.checked_since_edit = False, False

    def observe(self, tool: str, args: dict) -> None:
        if tool in _EDIT_TOOLS and not str(args.get("path", "")).rsplit("/", 1)[-1].startswith("scratch_"):
            self.edited, self.checked_since_edit = True, False
        elif _is_check(tool, args):
            self.checked_since_edit = True

    def intercept(self, tool: str) -> str | None:
        """Observation to return instead of executing `submit`, or None to let it through."""
        if (self.enabled and not self.used and tool == "submit" and self.edited
                and not self.checked_since_edit):
            self.used = True
            return SUBMIT_NUDGE
        return None


def build_task(problem_statement: str, test_cmd: str | None = None) -> str:
    task = f"<issue>\n{problem_statement.strip()}\n</issue>"
    if test_cmd:
        task += f"\n\nRun the test suite with: `{test_cmd}`"
    return task


@dataclass
class AgentResult:
    trajectory: list[Step]
    patch: str
    submitted: bool


def run_agent(policy: Policy, env: RepoEnv, problem_statement: str,
              test_cmd: str | None = None, max_steps: int = 40,
              on_step: Callable[[int, Step], None] | None = None,
              options: AgentOptions | None = None) -> AgentResult:
    """`on_step(i, step)` is called after every step (progress reporting); raising from it
    aborts the episode (cancellation). `options` defaults to the policy's `agent_options`
    (the student sets them from config), else plain AgentOptions()."""
    opts = options or getattr(policy, "agent_options", None) or AgentOptions()
    gate = SubmitGate(opts.submit_gate)
    policy.reset(opts.system_prompt(), build_task(problem_statement, test_cmd))
    trajectory: list[Step] = []
    obs: str | None = None
    for i in range(max_steps):
        thought, action = policy.act(obs)
        nudge = gate.intercept(action.tool)
        obs = nudge if nudge is not None else env.execute(action)
        gate.observe(action.tool, action.args or {})
        if not env.done:
            obs += steps_left_notice(max_steps - i - 1, opts.step_notice)
        trajectory.append(Step(thought, action, obs))
        if on_step is not None:
            on_step(i, trajectory[-1])
        if env.done:
            break
    return AgentResult(trajectory, env.diff(), env.done)
