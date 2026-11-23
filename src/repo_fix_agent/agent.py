"""The agent loop: plug any Policy onto any repo."""
from __future__ import annotations

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
    "Not submitted yet. You haven't checked your change since the last edit. Before submitting: "
    "list the behaviours the issue implies (including edge cases and other call paths that share the "
    "code you changed), check each one with a quick scratch_*.py script or command, and run the test "
    "suite. If everything holds, call `submit` again.")


@dataclass
class AgentOptions:
    """Inference scaffold options. The student's come from configs/models.yaml (student.serving)
    and are reused for SFT rendering, RL and distillation so training matches inference."""
    spec_check: bool = False     # extend the system prompt with the behaviour/edge-case checklist
    submit_gate: bool = False    # bounce the first `submit` that follows an unverified edit

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT + (SPEC_CHECK if self.spec_check else "")


def student_options(path: str = "configs/models.yaml") -> AgentOptions:
    from .policies import StudentServingConfig
    try:
        return StudentServingConfig.load(path).agent_options()
    except FileNotFoundError:
        return AgentOptions()


_EDIT_TOOLS = {"str_replace", "create_file", "insert", "undo_edit"}


def _is_check(tool: str, args: dict) -> bool:
    """A step that exercises code: running tests, a script, or python."""
    if tool == "run_tests":
        return True
    if tool == "bash":
        cmd = str(args.get("command", ""))
        return any(k in cmd for k in ("pytest", "python", "unittest", "tox", "make test"))
    return False


class SubmitGate:
    """Bounces at most one `submit` per episode when the last edit hasn't been checked since."""

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
        trajectory.append(Step(thought, action, obs))
        if on_step is not None:
            on_step(i, trajectory[-1])
        if env.done:
            break
    return AgentResult(trajectory, env.diff(), env.done)
