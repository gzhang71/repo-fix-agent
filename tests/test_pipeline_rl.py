import argparse
import copy
import json
from pathlib import Path

import pytest

from repo_fix_agent import policies
from repo_fix_agent.grader import GradeResult
from repo_fix_agent.pipeline import rl
from repo_fix_agent.pipeline.verifier import ExcludedInstanceError
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action, load_demos, save_demos

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/demos/seed.jsonl"
SP = rl.RewardConfig().step_penalty


@pytest.fixture(autouse=True)
def _restore_sandbox_warning_flag(monkeypatch):
    # grader.grade uses the "auto" backend, which flips sandbox's warn-once flag; restore it so
    # test_sandbox's fallback-warning test is unaffected by test ordering.
    from repo_fix_agent import sandbox
    monkeypatch.setattr(sandbox, "_warned_fallback", getattr(sandbox, "_warned_fallback", False))


def _demo(iid="seed-001-pagination-offset"):
    demos = load_demos(SEED)
    return copy.deepcopy(next((d for d in demos if d.instance_id == iid), demos[0]))


def _script(demo, start=0):
    return [(s.thought, s.action) for s in demo.trajectory[start:]]


def _edit_step(demo):
    return next(i for i, s in enumerate(demo.trajectory) if s.action.tool == "str_replace")


def _run(demo, steps, prefix=0, max_steps=40):
    env = rl.RepoFixEnv(ROOT, max_steps=max_steps, backend="local")
    return rl.run_episode(ScriptedPolicy(steps), env, demo, prefix)


def test_resolved_reward():
    d = _demo()
    ep = _run(d, _script(d))
    assert ep.resolved and ep.submitted
    assert ep.reward == pytest.approx(1 - SP * len(d.trajectory))
    assert ep.breakdown["test_infra_files"] == [] and ep.breakdown["total"] == ep.reward
    assert ep.grade["fail_to_pass"] and all(ep.grade["fail_to_pass"].values())


def test_unresolved_reward():
    ep = _run(_demo(), [("give up", Action("submit", {}))])
    assert not ep.resolved and ep.reward <= 0
    assert ep.reward == pytest.approx(-SP)


def test_test_infra_penalty_and_flag(caplog):
    d = _demo()
    steps = [("tamper", Action("create_file", {"path": "conftest.py", "content": "# harmless\n"}))] + _script(d)
    with caplog.at_level("WARNING"):
        ep = _run(d, steps)
    assert ep.resolved
    assert ep.breakdown["test_infra_files"] == ["conftest.py"]
    assert ep.breakdown["resolved_with_test_infra_edit"] is True
    assert ep.reward == pytest.approx(1 - SP * len(steps) - 1.0)
    assert "test infra" in caplog.text


def test_tool_error_format_penalty():
    d = _demo()
    bad = ("oops", Action("str_replace", {"path": "webapp/pagination.py", "old_str": "NOPE", "new_str": "x"}))
    ep = _run(d, [bad] + _script(d))
    assert ep.resolved and ep.breakdown["n_tool_errors"] == 1
    assert ep.reward == pytest.approx(1 - SP * (len(d.trajectory) + 1) - 0.05)


class RecordingPolicy(ScriptedPolicy):
    def preload(self, steps):
        self.preloaded = steps


def test_prefix_replay_and_preload_continuation():
    d = _demo()
    t = _edit_step(d) + 1  # prefix includes the engineer's edit
    env = rl.RepoFixEnv(ROOT, backend="local")
    task, info = env.reset(d, t)
    assert info["prefix_steps"] == t and len(info["prefix"]) == t
    assert info["last_observation"].startswith("Edited")
    assert "<issue>" in task
    env.close()

    pol = RecordingPolicy(_script(d, t))
    ep = rl.run_episode(pol, rl.RepoFixEnv(ROOT, backend="local"), d, t)
    assert len(pol.preloaded) == t
    assert ep.resolved and ep.prefix_steps == t
    assert ep.reward == pytest.approx(1 - SP * (len(d.trajectory) - t))  # prefix steps are not penalized
    assert ep.step_loss_mask == [False] * t + [True] * (len(d.trajectory) - t)

    # prefix stopping before the edit: a policy that only submits does not resolve
    ep2 = _run(d, [("submit", Action("submit", {}))], prefix=_edit_step(d))
    assert not ep2.resolved


def test_prefix_never_replays_submit():
    d = _demo()
    assert all(s.action.tool != "submit" for s in rl.prefix_of(d, 999))


def test_max_steps_termination():
    d = _demo()
    view = ("look", Action("view_file", {"path": "webapp/pagination.py"}))
    env = rl.RepoFixEnv(ROOT, max_steps=3, backend="local")
    env.reset(d)
    rewards, dones = [], []
    for _ in range(3):
        _, r, done, info = env.step(view[1], view[0])
        rewards.append(r)
        dones.append(done)
    assert dones == [False, False, True] and rewards[:2] == [0.0, 0.0]
    assert info["truncated"] and not info["resolved"]
    assert rewards[2] == pytest.approx(-3 * SP)
    with pytest.raises(RuntimeError):
        env.step(view[1])


def test_advantage_math():
    r = [1.0, 0.0, 0.0, 1.0]
    g = rl.grpo_advantages(r)
    assert g == pytest.approx([1, -1, -1, 1], abs=1e-4)
    assert rl.grpo_advantages([0.5, 0.5]) == pytest.approx([0, 0])
    rloo = rl.rloo_advantages([1.0, 0.0, 0.5])
    assert rloo == pytest.approx([1 - 0.25, 0 - 0.75, 0.5 - 0.5])
    assert rl.rloo_advantages([1.0]) == [0.0]


def _alt_factory(demo, pattern):
    calls = {"i": 0}

    def f():
        good = pattern[calls["i"] % len(pattern)]
        calls["i"] += 1
        return ScriptedPolicy(_script(demo) if good else [("give up", Action("submit", {}))])
    return f


def test_rollout_group_and_export():
    d = _demo()
    eps = rl.rollout_group(_alt_factory(d, [True, False]), d, group_size=2, root=ROOT, backend="local")
    assert [e.resolved for e in eps] == [True, False]
    assert eps[0].advantage_grpo > 0 > eps[1].advantage_grpo
    assert eps[0].advantage_rloo == pytest.approx(eps[0].reward - eps[1].reward)
    assert {e.group_id for e in eps} == {f"{d.instance_id}@p0"}
    rec = rl.episode_to_record(eps[0], d)
    assert set(rec) == {"instance_id", "group_id", "messages", "loss_mask", "reward", "advantage_grpo",
                        "advantage_rloo", "prefix_steps", "resolved"}
    assert len(rec["loss_mask"]) == len(rec["messages"])
    assert [m["role"] for m in rec["messages"][:2]] == ["system", "user"]
    assert all(mask == (m["role"] == "assistant") for m, mask in zip(rec["messages"], rec["loss_mask"]))
    json.dumps(rec)


def test_export_masks_prefix():
    d = _demo()
    t = _edit_step(d) + 1
    eps = rl.rollout_group(lambda: ScriptedPolicy(_script(d, t)), d, group_size=1, prefix_steps=t,
                           root=ROOT, backend="local")
    rec = rl.episode_to_record(eps[0], d)
    asst = [mask for m, mask in zip(rec["messages"], rec["loss_mask"]) if m["role"] == "assistant"]
    assert asst == [False] * t + [True] * (len(d.trajectory) - t)
    assert rec["prefix_steps"] == t and rec["resolved"]


def test_exclude_ids_refused():
    d = _demo()
    with pytest.raises(ExcludedInstanceError):
        rl.rollout_group(lambda: ScriptedPolicy([]), d, 1, root=ROOT, exclude_ids={d.instance_id})


def test_estimate_pass_at_k_with_fake_grader():
    d = _demo()
    fake = lambda demo, patch, root: GradeResult(bool(patch.strip()))  # noqa: E731
    rates = rl.estimate_pass_at_k(_alt_factory(d, [True, False, False]), [d], k=3, root=ROOT, backend="local",
                                  grade_fn=fake)
    assert rates == {d.instance_id: 1}


def test_curriculum_and_anneal():
    d = _demo()
    rates = {"a": 3, "b": 0, "c": 8, d.instance_id: 0, "e": 0.5}
    cur = rl.build_curriculum(rates, k=8, demos=[d])
    assert cur["informative"] == ["a", "e"] and cur["solved"] == ["c"]
    assert set(cur["hard"]) == {"b", d.instance_id}
    phases = [s["phase"] for s in cur["schedule"]]
    assert phases == sorted(phases) and all(s["instance_id"] != "c" for s in cur["schedule"])
    hard_d = next(s for s in cur["schedule"] if s["instance_id"] == d.instance_id)
    n = len(rl.prefix_of(d, 999))
    assert hard_d["prefix_steps"] == round(0.6 * n)
    assert rl.anneal_prefix(4, 0.0) == 4
    assert rl.anneal_prefix(4, 0.6) == 3
    assert rl.anneal_prefix(4, 1.0) == 2
    assert rl.anneal_prefix(1, 1.0) == 0


def _parser():
    p = argparse.ArgumentParser()
    rl.register(p.add_subparsers(dest="cmd", required=True))
    return p


def test_cli_rl_rollouts(tmp_path, monkeypatch, capsys):
    d = _demo()
    inp = tmp_path / "in.jsonl"
    save_demos([d], inp)
    monkeypatch.setattr(policies, "make_policy", lambda m: ScriptedPolicy(_script(d)))
    out = tmp_path / "batch.jsonl"
    a = _parser().parse_args(["rl-rollouts", str(inp), "--model", "student", "--group-size", "1", "--out", str(out),
                              "--root", str(ROOT), "--backend", "local"])
    assert a.fn(a) == 0
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["resolved"] and recs[0]["prefix_steps"] == 0

    # curriculum: pass==0 instance gets an auto prefix; the scripted policy continues from it
    t = round(0.6 * len(rl.prefix_of(d, 999)))
    monkeypatch.setattr(policies, "make_policy", lambda m: ScriptedPolicy(_script(d, t)))
    pr = tmp_path / "pass.json"
    pr.write_text(json.dumps({d.instance_id: 0}))
    a = _parser().parse_args(["rl-rollouts", str(inp), "--group-size", "1", "--out", str(out), "--root", str(ROOT),
                              "--backend", "local", "--pass-rates", str(pr)])
    assert a.fn(a) == 0
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert recs[0]["prefix_steps"] == t and recs[0]["resolved"]

    excl = tmp_path / "frozen.txt"
    excl.write_text(d.instance_id + "\n")
    a = _parser().parse_args(["rl-rollouts", str(inp), "--out", str(out), "--exclude-ids", str(excl)])
    assert a.fn(a) == 2
