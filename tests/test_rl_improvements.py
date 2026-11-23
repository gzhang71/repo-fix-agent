"""RL improvements: shaped reward, verified-repro bonus, dynamic sampling, steps-left notice. Offline."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from repo_fix_agent import agent as A
from repo_fix_agent.env import RepoEnv
from repo_fix_agent.pipeline import rl as RL
from repo_fix_agent.pipeline import rl_rewards as RR
from repo_fix_agent.pipeline import training as T
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action, load_demos

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("REPOFIX_SANDBOX", "local")


@pytest.fixture(scope="module")
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


# ---- 1. shaped credit on the hidden tests --------------------------------------------------------
def test_shaped_credit_and_ordering():
    g = lambda f2p, p2p: {"fail_to_pass": f2p, "pass_to_pass": p2p, "error": None}  # noqa: E731
    assert RR.shaped_credit(False, g({"a": True, "b": False}, {"x": True})) == {"partial_credit": 0.25, "p2p_penalty": 0.0}
    assert RR.shaped_credit(False, g({"a": True}, {"x": False, "y": True}))["p2p_penalty"] == -0.25
    assert RR.shaped_credit(True, g({"a": True}, {"x": True})) == {"partial_credit": 0.0, "p2p_penalty": 0.0}
    assert RR.shaped_credit(False, {"error": "patch failed to apply"}) == {"partial_credit": 0.0, "p2p_penalty": 0.0}

    best_unresolved, _ = RL.compute_reward(False, 5, 0, "", grade=g({"a": True, "b": False}, {"x": True}),
                                           repro={"valid_repro": True})
    worst_resolved, _ = RL.compute_reward(True, 40, 3, "", grade=g({"a": True, "b": True}, {"x": True}))
    assert best_unresolved < worst_resolved                       # resolved always ranks first
    partial, bd = RL.compute_reward(False, 5, 0, "", grade=g({"a": True, "b": False}, {}))
    zero, _ = RL.compute_reward(False, 5, 0, "", grade=g({"a": False, "b": False}, {}))
    assert partial > zero and bd["partial_credit"] == 0.25
    broke, _ = RL.compute_reward(False, 5, 0, "", grade=g({"a": True, "b": False}, {"x": False}))
    assert broke < partial                                         # breaking a passing test costs
    binary, bd = RL.compute_reward(False, 5, 0, "", RL.RewardConfig(reward="binary"),
                                   grade=g({"a": True, "b": False}, {"x": False}))
    assert bd["partial_credit"] == 0 and bd["p2p_penalty"] == 0 and binary == pytest.approx(-0.01)


# ---- 2. verified repro bonus -----------------------------------------------------------------------
def test_repro_check_rules():
    ok = RR.repro_check({"scratch_a.py": 0}, {"scratch_a.py": (1, "AssertionError: expected [3, 4]")})
    assert ok.valid
    assert not RR.repro_check({"s.py": 0}, {"s.py": (0, "")}).valid                       # passes on base: no repro
    assert not RR.repro_check({"s.py": 1}, {"s.py": (1, "AssertionError")}).valid          # still fails after the fix
    trivial = RR.repro_check({"s.py": 0}, {"s.py": (1, "ModuleNotFoundError: No module named 'x'")})
    assert not trivial.valid and trivial.scripts[0].base_trivial                           # didn't even run
    assert RR.repro_check({"s.py": 0}, {"s.py": (1, "AttributeError: no attribute 'fmt'")}).valid


REPRO = ("from webapp.pagination import paginate\n"
         "assert paginate(list(range(10)), 1, 3) == [0, 1, 2], paginate(list(range(10)), 1, 3)\n"
         "assert paginate(list(range(10)), 2, 3) == [3, 4, 5]\nprint('ok')\n")


def _gold_edit(pag):
    return next(s.action for s in pag.trajectory if s.action.tool == "str_replace")


def test_env_awards_verified_repro(pag, monkeypatch):
    steps = [("repro", Action("create_file", {"path": "scratch_page.py", "content": REPRO})),
             ("fix", _gold_edit(pag)),
             ("check", Action("bash", {"command": "python scratch_page.py"})),
             ("done", Action("submit", {}))]
    with RL.RepoFixEnv(ROOT) as env:
        ep = RL.run_episode(ScriptedPolicy(steps), env, pag)
    assert ep.resolved and "scratch_page.py" not in ep.patch
    rep = ep.breakdown["repro"]
    assert rep["valid_repro"] and rep["runs"][0]["base_rc"] != 0 and rep["runs"][0]["patched_rc"] == 0
    assert ep.breakdown["repro_bonus"] == 0.1 and ep.reward > 1.0

    bogus = [("repro", Action("create_file", {"path": "scratch_x.py", "content": "import not_a_module\n"})),
             ("fix", _gold_edit(pag)), ("done", Action("submit", {}))]
    monkeypatch.setattr(A.SubmitGate, "intercept", lambda self, tool: None)   # isolate the reward from the gate
    with RL.RepoFixEnv(ROOT, reward_config=RL.RewardConfig()) as env:
        ep2 = RL.run_episode(ScriptedPolicy(bogus), env, pag)
    assert ep2.resolved and ep2.breakdown["repro_bonus"] == 0.0 and not ep2.breakdown["repro"]["valid_repro"]
    rec = RL.episode_to_record(ep, pag)
    assert rec["valid_repro"] and "repro" not in rec["reward_breakdown"]


# ---- 4. dynamic sampling -----------------------------------------------------------------------------
def _ep(r):
    return NS(reward=r, resolved=r >= 1, breakdown={"resolved_with_test_infra_edit": False})


def test_informative_and_dynamic_sampling():
    assert not RL.informative([_ep(1), _ep(1), _ep(1)]) and not RL.informative([_ep(0), _ep(0)])
    assert RL.informative([_ep(1), _ep(0)]) and RL.informative([_ep(0.25), _ep(0)])     # shaped breaks ties
    rewards = {"a": [1, 1], "b": [0, 1], "c": [0, 0], "d": [0.5, 0], "e": [1, 0]}
    plan = [(NS(instance_id=k), 0) for k in rewards]
    seen = []
    kept, stats = RL.dynamic_sample(plan, lambda d, t: [_ep(r) for r in rewards[d.instance_id]],
                                    target_groups=2, batch=1, on_group=lambda d, t, e, ok: seen.append((d.instance_id, ok)))
    assert [d.instance_id for d, _, _ in kept] == ["b", "d"]
    assert stats["dropped_uniform"] == 2 and stats["tried"] == 4 and stats["plan_left"] == 1
    assert seen == [("a", False), ("b", True), ("c", False), ("d", True)]
    kept, stats = RL.dynamic_sample(plan, lambda d, t: [_ep(r) for r in rewards[d.instance_id]], batch=2)
    assert len(kept) == 3 and stats["plan_left"] == 0


# ---- 5. steps-left notice ----------------------------------------------------------------------------
def test_steps_left_notice_text():
    assert A.steps_left_notice(9, 8) == "" and A.steps_left_notice(3, 0) == ""
    assert "3 steps left" in A.steps_left_notice(3, 8)
    assert "last action" in A.steps_left_notice(1, 8) and A.steps_left_notice(0, 8) == ""


def test_notice_in_run_agent_and_rl_env(pag):
    looks = [("look", Action("view_file", {"path": "webapp/pagination.py"}))] * 6
    with RepoEnv(ROOT / pag.repo) as env:
        res = A.run_agent(ScriptedPolicy(looks), env, pag.problem_statement, pag.test_cmd, max_steps=6,
                          options=A.AgentOptions(step_notice=3))
    obs = [s.observation for s in res.trajectory]
    assert not any("steps left" in o or "step left" in o for o in obs[:2])
    assert "3 steps left" in obs[2] and "2 steps left" in obs[3] and "last action" in obs[4]
    assert "left" not in obs[5].split("\n")[-1]                    # nothing after the final action

    with RL.RepoFixEnv(ROOT, max_steps=6) as env:
        env.options = A.AgentOptions(step_notice=3)
        ep = RL.run_episode(ScriptedPolicy(looks), env, pag)
    assert [("left" in s.observation.split("\n")[-1]) for s in ep.trajectory] == [False, False, True, True, True, False]
    assert ep.breakdown["truncated"] and not ep.submitted


def test_student_config_enables_notice():
    from repo_fix_agent.policies import StudentServingConfig
    cfg = StudentServingConfig.load(ROOT / "configs/models.yaml")
    assert cfg.step_notice == 8 and cfg.agent_options().step_notice == 8


def test_plan_uses_new_rl_settings(tmp_path):
    st = {s.name: s for s in T.plan(tmp_path / "r", tmp_path / "s")}
    assert "--dynamic-sampling" in st["rl-rollouts"].argv and "shaped" in st["rl-rollouts"].argv
    assert "--clip-high 0.28" in st["train-rl"].external and "--loss-agg token" in st["train-rl"].external


def test_sft_rendering_adds_notice_once(pag):
    from repo_fix_agent import sft
    short = dataclasses.replace(pag, trajectory=pag.trajectory[:3])
    msgs = sft.render_messages(short, step_notice=8, max_steps=3)
    tool_obs = [m["content"] for m in msgs if m["role"] == "tool"]
    assert "2 steps left" in tool_obs[0] and "last action" in tool_obs[1]
    recorded = dataclasses.replace(short, trajectory=[dataclasses.replace(s, observation=o)
                                                      for s, o in zip(short.trajectory, tool_obs)])
    again = [m["content"] for m in sft.render_messages(recorded, step_notice=8, max_steps=3) if m["role"] == "tool"]
    assert again == tool_obs                                       # already present: not doubled
    assert sft.render_messages(short)[3]["content"] == short.trajectory[0].observation   # off by default
