"""§3e preference data: trajectory / patch / step pairs, KTO, length balancing, exclusion, CLI."""
import argparse
import copy
import json

import pytest

from repo_fix_agent import sandbox as sb
from repo_fix_agent.pipeline import preferences as P
from repo_fix_agent.pipeline.sft_data import read_jsonl
from repo_fix_agent.schema import Action, Step, save_demos

try:
    from test_pipeline_sft import PATCH, make_demo
except ImportError:  # tests/ became a package
    from tests.test_pipeline_sft import PATCH, make_demo

BAD_PATCH = PATCH.replace("(page - 1) * per_page", "page * per_page - 1")


@pytest.fixture(autouse=True)
def _local_sandbox(monkeypatch):
    monkeypatch.setattr(sb, "_docker_available", lambda: False)
    # once-per-process warning flags: restore them so test_sandbox's log assertions stay order-independent
    monkeypatch.setattr(sb, "_warned_fallback", sb._warned_fallback)
    monkeypatch.setattr(sb, "_warned_network", sb._warned_network)


@pytest.fixture
def demo(tmp_path):
    return make_demo(tmp_path)


def rollout(d, k, resolved, source="student:s", truncate=None, diverge_at=None, patch=None):
    r = copy.deepcopy(d)
    r.instance_id = f"{d.instance_id}#r{k}"
    r.metadata = {"source": source, "resolved": resolved, "parent_id": d.instance_id}
    if diverge_at is not None:  # wrong edit + premature submit from step `diverge_at`
        r.trajectory = r.trajectory[:diverge_at] + [
            Step("Guess the bug is in the bound.", Action("str_replace", {
                "path": "webapp/pagination.py", "old_str": "start = page * per_page",
                "new_str": "start = page * per_page - 1"}), "Edited webapp/pagination.py."),
            Step("Done.", Action("submit", {}), "Submitted.")]
    if truncate is not None:
        r.trajectory = r.trajectory[:truncate]
    if patch is not None:
        r.patch = patch
    elif not resolved:
        r.patch = BAD_PATCH
    return r


def test_trajectory_pairs_prefer_on_policy(demo):
    rs = [rollout(demo, 0, True), rollout(demo, 1, False, diverge_at=2),
          rollout(demo, 2, True, source="teacher:t"), rollout(demo, 3, False, source="teacher:t", diverge_at=1)]
    pairs = P.trajectory_pairs(rs)
    assert len(pairs) == 1 and pairs[0]["meta"]["on_policy"]
    p = pairs[0]
    assert [m["role"] for m in p["prompt"]] == ["system", "user"]
    assert p["chosen"][0]["role"] == "assistant" and p["chosen"] != p["rejected"]
    assert p["meta"]["chosen_len"] > p["meta"]["rejected_len"] > 0
    # no student pair available -> fall back to any source
    mixed = P.trajectory_pairs([rs[2], rs[1]])
    assert len(mixed) == 1 and not mixed[0]["meta"]["on_policy"]
    # different parents never pair
    other = rollout(demo, 4, False)
    other.metadata["parent_id"] = "other"
    assert P.trajectory_pairs([rs[0], other]) == []


def test_patch_pairs_gold_vs_failed(demo, tmp_path):
    rs = [rollout(demo, 0, False), rollout(demo, 1, False),  # duplicate bad patch -> one pair
          rollout(demo, 2, False, patch=""), rollout(demo, 3, True)]
    pairs = P.patch_pairs([demo], rs, tmp_path)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["chosen"] == [{"role": "assistant", "content": demo.patch}]
    assert p["rejected"][0]["content"] == BAD_PATCH
    assert "def paginate" in p["prompt"][1]["content"] and p["type"] == "patch"
    assert all(m["role"] != "tool" for m in p["prompt"])  # single-turn, no trajectory splicing


def test_step_pairs_at_first_divergence(demo):
    good, bad = rollout(demo, 0, True), rollout(demo, 1, False, diverge_at=2)
    (p,) = P.step_pairs([good, bad])
    assert p["meta"]["prefix_len"] == 2
    assert [m["role"] for m in p["prompt"]] == ["system", "user"] + ["assistant", "tool"] * 2
    chosen_args = json.loads(p["chosen"][0]["tool_calls"][0]["function"]["arguments"])
    rejected_args = json.loads(p["rejected"][0]["tool_calls"][0]["function"]["arguments"])
    assert chosen_args["new_str"] == "start = (page - 1) * per_page"
    assert rejected_args["new_str"] == "start = page * per_page - 1"
    # truncated (prefix-of-resolved) unresolved rollout has no next action -> no pair
    assert P.step_pairs([good, rollout(demo, 2, False, truncate=3)]) == []


def test_kto_labels(demo):
    ex = P.kto_examples([rollout(demo, 0, True), rollout(demo, 1, False, diverge_at=1),
                         rollout(demo, 2, None)])
    assert [e["label"] for e in ex] == [True, False]
    assert ex[0]["completion"][0]["role"] == "assistant" and len({e["id"] for e in ex}) == 2


def test_length_balancing():
    def pair(c, r):
        return {"meta": {"chosen_len": c, "rejected_len": r}}

    pairs = [pair(10, 100), pair(20, 100), pair(50, 60), pair(90, 100), pair(120, 100), pair(70, 70)]
    before = P.length_stats(pairs)
    assert before["frac_chosen_shorter"] == pytest.approx(4 / 6)
    kept = P.balance_lengths(pairs)
    after = P.length_stats(kept)
    shorter = sum(p["meta"]["chosen_len"] < p["meta"]["rejected_len"] for p in kept)
    longer = sum(p["meta"]["chosen_len"] > p["meta"]["rejected_len"] for p in kept)
    assert shorter == longer == 1 and pair(70, 70) in kept
    assert abs(after["mean_chosen"] - after["mean_rejected"]) < abs(before["mean_chosen"] - before["mean_rejected"])


def test_build_preferences_and_exclude(demo, tmp_path):
    rs = [rollout(demo, 0, True), rollout(demo, 1, False, diverge_at=2)]
    out = P.build_preferences(rs, [demo], tmp_path, balance=True)
    assert {p["type"] for p in out["dpo"]} <= {"trajectory", "patch"}
    assert out["stats"]["kto"] == {"n": 2, "desirable": 1} and "balanced" in out["stats"]
    with pytest.raises(ValueError, match="excluded"):
        P.build_preferences(rs, [], tmp_path, exclude_ids={demo.instance_id})


def test_cli_prefs(demo, tmp_path, capsys):
    r1, r2, dm = tmp_path / "r1.jsonl", tmp_path / "r2.jsonl", tmp_path / "d.jsonl"
    save_demos([rollout(demo, 0, True)], r1)
    save_demos([rollout(demo, 1, False, diverge_at=2)], r2)
    save_demos([demo], dm)
    outs = {k: tmp_path / f"{k}.jsonl" for k in ("dpo", "kto", "steps")}
    p = argparse.ArgumentParser()
    P.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["prefs", "--rollouts", str(r1), str(r2), "--demos", str(dm), "--root", str(tmp_path),
                      "--out-dpo", str(outs["dpo"]), "--out-kto", str(outs["kto"]),
                      "--out-steps", str(outs["steps"])])
    assert a.fn(a) == 0
    assert sorted(p["type"] for p in read_jsonl(outs["dpo"])) == ["patch", "trajectory"]
    assert len(read_jsonl(outs["kto"])) == 2 and len(read_jsonl(outs["steps"])) == 1
    assert json.loads(capsys.readouterr().out)["trajectory"]["n"] == 1
    ex = tmp_path / "frozen.json"
    ex.write_text(json.dumps({"instances": [demo.instance_id]}))
    a = p.parse_args(["prefs", "--rollouts", str(r1), "--out-dpo", str(outs["dpo"]), "--exclude-ids", str(ex)])
    assert a.fn(a) == 2  # frozen-test violation -> clean error, exit 2
