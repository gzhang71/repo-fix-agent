"""The agent loop: plug any Policy onto any repo."""
from __future__ import annotations

from dataclasses import dataclass

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
              test_cmd: str | None = None, max_steps: int = 40) -> AgentResult:
    policy.reset(SYSTEM_PROMPT, build_task(problem_statement, test_cmd))
    trajectory: list[Step] = []
    obs: str | None = None
    for _ in range(max_steps):
        thought, action = policy.act(obs)
        obs = env.execute(action)
        trajectory.append(Step(thought, action, obs))
        if env.done:
            break
    return AgentResult(trajectory, env.diff(), env.done)
