"""Difficulty-aware weighting (pipeline/difficulty.py) and its consumers. Offline."""
import argparse
import dataclasses
import json

import pytest

from repo_fix_agent.pipeline import difficulty as D
from repo_fix_agent.pipeline import sft_data as S
from repo_fix_agent.schema import load_demos

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


def _evals(spec):   # {instance: [True, False, ...]}
    return [{"instance_id": k, "sample": i, "resolved": v, "error": None}
            for k, vs in spec.items() for i, v in enumerate(vs)]


def test_buckets_and_weights():
    student = _evals({"gap": [0, 0, 0, 0], "mid": [1, 0, 0, 1], "easy": [1, 1, 1, 1], "dead": [0, 0, 0, 0]})
    teacher = _evals({"gap": [1], "mid": [1], "easy": [1], "dead": [0]})
    w = D.compute_weights(student, teacher)
    inst = w["instances"]
    assert {k: v["bucket"] for k, v in inst.items()} == {
        "gap": "gap", "mid": "learning", "easy": "mastered", "dead": "unsolved"}
    assert inst["gap"]["weight"] == 1.0 and inst["mid"]["weight"] == 0.5
    assert inst["easy"]["weight"] == 0.2 and inst["dead"]["weight"] == 0.0
    assert w["summary"]["instances"] == 4 and w["summary"]["learning"] == 1


def test_errors_not_counted_and_unknown_teacher_is_gap():
    student = _evals({"a": [0, 1]}) + [{"instance_id": "a", "resolved": False, "error": "APIError: x"}]
    w = D.compute_weights(student)["instances"]["a"]
    assert w["samples"] == 2 and w["p"] == 0.5
    assert D.compute_weights(_evals({"b": [0, 0]}))["instances"]["b"]["bucket"] == "gap"


def test_rollout_demos_counted_by_parent():
    recs = [{"instance_id": f"x#s{i}", "metadata": {"parent_id": "x", "resolved": i == 0}} for i in range(4)]
    assert D.pass_counts(recs) == {"x": (1, 4)}


def test_apply_weights_drop_cap_and_aux_untouched():
    w = {"a": {"p": 0.0, "weight": 1.0}, "b": {"p": 1.0, "weight": 0.2}, "z": {"p": 0.0, "weight": 0.0}}
    recs = ([{"id": f"a#{i}", "weight": 2.0, "source": "teacher"} for i in range(3)]
            + [{"id": "b", "weight": 1.0, "source": "engineer", "meta": {"parent_id": "b"}},
               {"id": "z", "weight": 1.0, "source": "teacher"},
               {"id": "q", "weight": 1.0, "source": "teacher"},
               {"id": "aux1", "weight": 1.0, "source": "aux:localize", "meta": {"instance_id": "z"}}])
    out, stats = D.apply_instance_weights(recs, w, unknown_weight=0.7, max_per_instance=3.0)
    by = {r["id"]: r for r in out}
    assert "z" not in by and stats["dropped_zero_weight"] == 1
    assert sum(by[f"a#{i}"]["weight"] for i in range(3)) == pytest.approx(3.0, abs=1e-3)   # 6.0 capped to 3.0
    assert by["b"]["weight"] == 0.2 and by["q"]["weight"] == 0.7 and by["aux1"]["weight"] == 1.0


def test_keep_for_teacher():
    w = {"easy": {"p": 1.0}, "mid": {"p": 0.5}}
    assert not D.keep_for_teacher("easy", w, 0.75)
    assert D.keep_for_teacher("mid", w, 0.75) and D.keep_for_teacher("unseen", w, 0.75)


def test_student_successes_shortest_first(pag):
    mk = lambda i, ok, n: dataclasses.replace(pag, instance_id=f"{pag.instance_id}#s{i}", trajectory=pag.trajectory[:n],
                                              metadata={"parent_id": pag.instance_id, "resolved": ok, "sample_idx": i})
    n = len(pag.trajectory)
    picked = D.select_student_successes([mk(0, True, n), mk(1, True, n - 1), mk(2, False, 1)])
    assert [d.metadata["sample_idx"] for d in picked] == [1]


def test_cli_end_to_end(pag, tmp_path, capsys):
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in _evals({pag.instance_id: [1, 0, 0, 0], "other": [1, 1]})))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    D.register(sub)
    S.register(sub)
    a = p.parse_args(["difficulty-weights", str(ev), "--out", str(tmp_path / "w.json"),
                      "--pass-rates-out", str(tmp_path / "pr.json")])
    assert a.fn(a) == 0
    assert json.loads((tmp_path / "pr.json").read_text()) == {pag.instance_id: 0.25, "other": 1.0}
    assert "learning" in capsys.readouterr().out

    demos = tmp_path / "demos.jsonl"
    demos.write_text(json.dumps(dataclasses.asdict(pag)) + "\n")
    rollouts = tmp_path / "rollouts.jsonl"
    own = dataclasses.replace(pag, instance_id=f"{pag.instance_id}#s0",
                              metadata={"parent_id": pag.instance_id, "resolved": True, "source": "student:s"})
    rollouts.write_text(json.dumps(dataclasses.asdict(own)) + "\n")
    out = tmp_path / "sft.jsonl"
    a = p.parse_args(["sft-mix", "--engineer", str(demos), "--student-rollouts", str(rollouts),
                      "--instance-weights", str(tmp_path / "w.json"), "--config", "none", "--out", str(out)])
    assert a.fn(a) == 0
    recs = S.read_jsonl(out)
    assert sorted(r["source"] for r in recs) == ["engineer_seed", "student-self"]
    assert all(r["weight"] == 0.75 and r["meta"]["parent_id"] == pag.instance_id for r in recs)
