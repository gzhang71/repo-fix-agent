"""§3g RL with execution reward: environment, reward, grouped rollouts, curriculum, batch export.

This module only *produces* rollout batches (with GRPO and RLOO advantages and loss masks) for an
external trainer (TRL / verl / OpenRLHF). It does not do gradient updates.

CLI:
  repo-fix rl-rollouts IN.jsonl --model M --group-size 8 --out batch.jsonl
      [--prefix-steps auto|N] [--pass-rates P.json] [--exclude-ids FILE] [--root .]
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agent import SubmitGate, build_task, student_options
from ..schema import Action, Demo, Step, load_demos
from .parallel import pmap
from .verifier import (ExcludedInstanceError, check_not_excluded, load_exclude_ids, parse_patch,
                       touches_test_infra)

log = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    step_penalty: float = 0.002
    format_penalty: float = 0.05
    test_infra_penalty: float = 1.0
    test_infra_patterns: tuple[str, ...] = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                                            ".github/", "noxfile.py")


def compute_reward(resolved: bool, n_steps: int, n_tool_errors: int, patch: str,
                   cfg: RewardConfig | None = None) -> tuple[float, dict[str, Any]]:
    cfg = cfg or RewardConfig()
    infra = touches_test_infra(parse_patch(patch)["files"], cfg.test_infra_patterns)
    bd = {
        "resolved": float(bool(resolved)),
        "step_penalty": -cfg.step_penalty * n_steps,
        "format_penalty": -cfg.format_penalty * n_tool_errors,
        "test_infra_penalty": -cfg.test_infra_penalty if infra else 0.0,
        "n_steps": n_steps,
        "n_tool_errors": n_tool_errors,
        "test_infra_files": infra,
        "resolved_with_test_infra_edit": bool(resolved and infra),
    }
    bd["total"] = bd["resolved"] + bd["step_penalty"] + bd["format_penalty"] + bd["test_infra_penalty"]
    return bd["total"], bd


def _repo_path(demo: Demo, root: str | Path) -> str:
    return demo.repo if "://" in demo.repo else str(Path(root) / demo.repo)


def prefix_of(demo: Demo, prefix_steps: int) -> list[Step]:
    """First `prefix_steps` engineer steps, never including the submit."""
    out = []
    for st in demo.trajectory[:prefix_steps]:
        if st.action.tool == "submit":
            break
        out.append(st)
    return out


class RepoFixEnv:
    """Gym-style episode over one instance. Reward is 0 until done, then
    resolved - step_penalty*steps - format_penalty*tool_errors - test_infra_penalty*[patch touches infra].
    Step/error counts cover policy steps only (the replayed engineer prefix is not penalized);
    `max_steps` bounds the total trajectory including the prefix, like the inference budget."""

    def __init__(self, root: str | Path = ".", max_steps: int = 40, reward_config: RewardConfig | None = None,
                 backend: str = "auto", grade_fn: Callable | None = None):
        from ..grader import grade

        self.root, self.max_steps, self.backend = root, max_steps, backend
        self.options = student_options()   # RL trains the student: its prompt + submit gate
        self.cfg = reward_config or RewardConfig()
        self.grade_fn = grade_fn or grade
        self.env = None
        self.demo: Demo | None = None
        self.trajectory: list[Step] = []
        self.n_prefix = 0
        self.done = True

    def reset(self, demo: Demo, prefix_steps: int = 0) -> tuple[str, dict[str, Any]]:
        from ..env import RepoEnv

        self.close()
        self.demo = demo
        self.env = RepoEnv(_repo_path(demo, self.root), demo.base_commit, backend=self.backend)
        replayed = []
        for st in prefix_of(demo, prefix_steps):
            replayed.append(Step(st.thought, st.action, self.env.execute(st.action)))
        self.trajectory, self.n_prefix, self.done = list(replayed), len(replayed), False
        task = build_task(demo.problem_statement, demo.test_cmd)
        self.gate = SubmitGate(self.options.submit_gate)
        for st in replayed:
            self.gate.observe(st.action.tool, st.action.args or {})
        return task, {"system": self.options.system_prompt(), "task": task, "instance_id": demo.instance_id,
                      "prefix": replayed, "prefix_steps": len(replayed),
                      "last_observation": replayed[-1].observation if replayed else None}

    def step(self, action: Action, thought: str = "") -> tuple[str, float, bool, dict[str, Any]]:
        if self.done or self.env is None:
            raise RuntimeError("episode is done; call reset()")
        nudge = self.gate.intercept(action.tool)   # same submit gate as inference (run_agent)
        obs = nudge if nudge is not None else self.env.execute(action)
        self.gate.observe(action.tool, action.args or {})
        self.trajectory.append(Step(thought, action, obs))
        submitted = self.env.done
        if not submitted and len(self.trajectory) < self.max_steps:
            return obs, 0.0, False, {}
        return obs, *self._finish(submitted)

    def _finish(self, submitted: bool) -> tuple[float, bool, dict[str, Any]]:
        patch = self.env.diff()
        self.close()
        self.done = True
        g = self.grade_fn(self.demo, patch, self.root)
        policy_steps = self.trajectory[self.n_prefix:]
        n_err = sum(s.observation.startswith("Error:") for s in policy_steps)
        reward, bd = compute_reward(g.resolved, len(policy_steps), n_err, patch, self.cfg)
        if bd["test_infra_files"]:
            log.warning("%s: patch touches test infra %s (resolved=%s)", self.demo.instance_id,
                        bd["test_infra_files"], g.resolved)
        info = {"patch": patch, "resolved": g.resolved, "submitted": submitted, "truncated": not submitted,
                "grade": {"fail_to_pass": g.fail_to_pass, "pass_to_pass": g.pass_to_pass, "error": g.error},
                "reward_breakdown": bd, "test_infra_hack": bd["resolved_with_test_infra_edit"],
                "trajectory": list(self.trajectory), "prefix_steps": self.n_prefix}
        return reward, True, info

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class Episode:
    instance_id: str
    group_id: str
    trajectory: list[Step]
    prefix_steps: int
    reward: float
    breakdown: dict[str, Any]
    resolved: bool
    patch: str
    submitted: bool = False
    sample_idx: int = 0
    grade: dict[str, Any] = field(default_factory=dict)
    advantage_grpo: float = 0.0
    advantage_rloo: float = 0.0

    @property
    def step_loss_mask(self) -> list[bool]:
        return [i >= self.prefix_steps for i in range(len(self.trajectory))]


def run_episode(policy, env: RepoFixEnv, demo: Demo, prefix_steps: int = 0) -> Episode:
    task, info = env.reset(demo, prefix_steps)
    policy.reset(info["system"], task)
    if info["prefix"] and hasattr(policy, "preload"):
        policy.preload(info["prefix"])
    obs, done, step_info = info["last_observation"], False, {}
    while not done:
        thought, action = policy.act(obs)
        obs, reward, done, step_info = env.step(action, thought)
    return Episode(demo.instance_id, f"{demo.instance_id}@p{info['prefix_steps']}", step_info["trajectory"],
                   info["prefix_steps"], reward, step_info["reward_breakdown"], step_info["resolved"],
                   step_info["patch"], step_info["submitted"], grade=step_info["grade"])


def grpo_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (std + eps) for r in rewards]


def rloo_advantages(rewards: Sequence[float]) -> list[float]:
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def rollout_group(policy_factory: Callable[[], Any], demo: Demo, group_size: int = 8, prefix_steps: int = 0,
                  root: str | Path = ".", max_steps: int = 40, reward_config: RewardConfig | None = None,
                  backend: str = "auto", exclude_ids: set[str] | None = None,
                  grade_fn: Callable | None = None, workers: int = 1, pool=None) -> list[Episode]:
    """`group_size` episodes on one instance with group-relative advantages. With `workers` > 1
    or a shared `pool`, episodes run concurrently, each in its own RepoFixEnv."""
    check_not_excluded([demo], exclude_ids, "RL training")

    def one(i: int) -> Episode:
        with RepoFixEnv(root, max_steps, reward_config, backend, grade_fn) as env:
            ep = run_episode(policy_factory(), env, demo, prefix_steps)
        ep.sample_idx = i
        return ep

    eps = pmap(one, range(group_size), workers, pool)
    rewards = [e.reward for e in eps]
    for e, g, r in zip(eps, grpo_advantages(rewards), rloo_advantages(rewards)):
        e.advantage_grpo, e.advantage_rloo = g, r
    return eps


# ---- curriculum -----------------------------------------------------------------------------
def estimate_pass_at_k(policy_factory: Callable[[], Any], demos: Sequence[Demo], k: int = 8,
                       root: str | Path = ".", max_steps: int = 40, backend: str = "auto",
                       exclude_ids: set[str] | None = None, grade_fn: Callable | None = None,
                       workers: int = 1) -> dict[str, int]:
    """Number of resolved samples out of k per instance."""
    return {d.instance_id: sum(e.resolved for e in rollout_group(
        policy_factory, d, k, 0, root, max_steps, backend=backend, exclude_ids=exclude_ids, grade_fn=grade_fn,
        workers=workers)) for d in demos}


def initial_prefix(demo: Demo | int, frac: float = 0.6) -> int:
    n = demo if isinstance(demo, int) else len(prefix_of(demo, len(demo.trajectory)))
    return max(1, round(frac * n)) if n else 0


def anneal_prefix(t: int, success_rate: float, target: float = 0.5, step: int = 1) -> int:
    """Shorten the given prefix as the student succeeds from it (faster when success is high)."""
    if success_rate >= 0.9:
        return max(0, t - 2 * step)
    if success_rate >= target:
        return max(0, t - step)
    return t


def _as_count(v: float, k: int) -> int:
    # ints are success counts; floats are fractions of k
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else round(float(v) * k)


def build_curriculum(pass_rates: Mapping[str, float], k: int = 8, demos: Sequence[Demo] | None = None,
                     prefix_frac: float = 0.6) -> dict[str, Any]:
    """Buckets: informative (0<pass<k), hard (pass==0, prefix-conditioned), solved (pass==k, dropped).
    Schedule: informative first (no prefix), then hard with prefix t ~= prefix_frac * engineer length."""
    by_id = {d.instance_id: d for d in demos or []}
    informative, hard, solved = [], [], []
    for iid, v in sorted(pass_rates.items()):
        c = _as_count(v, k)
        (solved if c >= k else hard if c <= 0 else informative).append(iid)
    schedule = [{"instance_id": i, "phase": 1, "prefix_steps": 0} for i in informative]
    schedule += [{"instance_id": i, "phase": 2,
                  "prefix_steps": initial_prefix(by_id[i], prefix_frac) if i in by_id else None} for i in hard]
    return {"k": k, "informative": informative, "hard": hard, "solved": solved, "schedule": schedule}


# ---- export ---------------------------------------------------------------------------------
def episode_to_record(ep: Episode, demo: Demo) -> dict[str, Any]:
    """`loss_mask` is aligned with `messages`: True only on assistant turns the policy generated."""
    from ..sft import render_messages

    msgs = render_messages(replace(demo, trajectory=ep.trajectory))
    mask, step_i = [], 0
    for m in msgs:
        if m["role"] == "assistant":
            mask.append(step_i >= ep.prefix_steps)
            step_i += 1
        else:
            mask.append(False)
    return {"instance_id": ep.instance_id, "group_id": ep.group_id, "messages": msgs, "loss_mask": mask,
            "reward": ep.reward, "advantage_grpo": ep.advantage_grpo, "advantage_rloo": ep.advantage_rloo,
            "prefix_steps": ep.prefix_steps, "resolved": ep.resolved}


def write_batch(records: Sequence[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- CLI ------------------------------------------------------------------------------------
def cmd_rl_rollouts(a) -> int:
    from .. import policies

    demos = load_demos(a.demos)
    exclude = load_exclude_ids(a.exclude_ids)
    try:
        check_not_excluded(demos, exclude, "RL training")
    except ExcludedInstanceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    by_id = {d.instance_id: d for d in demos}
    if a.pass_rates:
        cur = build_curriculum(json.loads(Path(a.pass_rates).read_text()), a.group_size, demos)
        print(f"curriculum: {len(cur['informative'])} informative, {len(cur['hard'])} hard (prefix), "
              f"{len(cur['solved'])} solved (dropped)", file=sys.stderr)
        plan = [(by_id[s["instance_id"]], s["prefix_steps"] or 0) for s in cur["schedule"] if s["instance_id"] in by_id]
    elif a.prefix_steps == "auto":
        plan = [(d, 0) for d in demos]
    else:
        plan = [(d, int(a.prefix_steps)) for d in demos]
    if a.prefix_steps not in ("auto", None) and a.pass_rates:
        plan = [(d, int(a.prefix_steps) if t else 0) for d, t in plan]
    records = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, a.workers), thread_name_prefix="episode") as episodes:
        groups = pmap(lambda dt: rollout_group(lambda: policies.make_policy(a.model), dt[0], a.group_size, dt[1],
                                               a.root, a.max_steps, backend=a.backend, exclude_ids=exclude,
                                               pool=episodes if a.workers > 1 else None),
                      plan, a.workers)
    for (demo, t), eps in zip(plan, groups):
        records += [episode_to_record(e, demo) for e in eps]
        print(json.dumps({"instance_id": demo.instance_id, "prefix_steps": t,
                          "resolved": sum(e.resolved for e in eps), "group_size": len(eps),
                          "mean_reward": sum(e.reward for e in eps) / max(1, len(eps)),
                          "test_infra_hacks": sum(e.breakdown["resolved_with_test_infra_edit"] for e in eps)}),
              file=sys.stderr)
    write_batch(records, a.out)
    print(f"wrote {len(records)} episodes to {a.out}")
    return 0


def register(subparsers) -> None:
    r = subparsers.add_parser("rl-rollouts", help="§3g grouped rollouts with execution reward for an RL trainer")
    r.add_argument("demos")
    r.add_argument("--model", default="student")
    r.add_argument("--group-size", type=int, default=8)
    r.add_argument("--out", default="runs/rl/batch.jsonl")
    r.add_argument("--prefix-steps", default="auto",
                   help="auto (curriculum prefix for pass==0 when --pass-rates is given, else 0) or N")
    r.add_argument("--pass-rates", default=None, help="JSON {instance_id: successes out of k (int) or fraction}")
    r.add_argument("--exclude-ids", default=None)
    r.add_argument("--root", default=".")
    r.add_argument("--max-steps", type=int, default=40)
    r.add_argument("--backend", default="auto")
    r.add_argument("--workers", type=int, default=1, help="concurrent episodes (threads)")
    r.set_defaults(fn=cmd_rl_rollouts)
