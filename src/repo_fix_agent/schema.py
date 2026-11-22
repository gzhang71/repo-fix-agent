"""Dataset schema for bug-fix demos.

One record = one bug instance (SWE-bench style) + one solving trajectory.
The same record type is used for:
  * engineer demos (the 10k human set, source="engineer"),
  * teacher rollouts (source="teacher:<model>"),
  * student rollouts during eval / RL (source="student:<model>").
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Action:
    tool: str                      # one of tools.TOOL_NAMES
    args: dict[str, Any]


@dataclass
class Step:
    thought: str                   # short rationale written before the action
    action: Action
    observation: str               # exact tool output the agent saw


@dataclass
class Demo:
    instance_id: str
    repo: str                      # local path (relative to dataset root) or git URL
    base_commit: str | None        # None for local seed repos
    problem_statement: str         # the issue text the agent receives
    test_cmd: str                  # how to run tests, e.g. "python -m pytest -q"
    test_patch: str                # hidden tests, applied only at grading time
    FAIL_TO_PASS: list[str]        # must fail before fix, pass after
    PASS_TO_PASS: list[str]        # must pass before and after (regression guard)
    patch: str                     # gold patch (source files only)
    trajectory: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Demo":
        steps = [
            Step(thought=s["thought"], action=Action(**s["action"]), observation=s["observation"])
            for s in d.get("trajectory", [])
        ]
        return cls(**{**d, "trajectory": steps})


def load_demos(path: str | Path) -> list[Demo]:
    return list(iter_demos(path))


def iter_demos(path: str | Path) -> Iterator[Demo]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Demo.from_dict(json.loads(line))


def save_demos(demos: Iterable[Demo], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in demos:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
