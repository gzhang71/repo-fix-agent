import argparse
import copy
import json
import random
from pathlib import Path

import pytest

from repo_fix_agent import policies
from repo_fix_agent.pipeline import verifier as V
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action, Step, load_demos, save_demos

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/demos/seed.jsonl"


@pytest.fixture(autouse=True)
def _restore_sandbox_warning_flag(monkeypatch):
    # grader.grade uses the "auto" backend, which flips sandbox's warn-once flag; restore it so
    # test_sandbox's fallback-warning test is unaffected by test ordering.
    from repo_fix_agent import sandbox
    monkeypatch.setattr(sandbox, "_warned_fallback", getattr(sandbox, "_warned_fallback", False))


def _demo(iid="seed-001-pagination-offset"):
    demos = load_demos(SEED)
    return copy.deepcopy(next((d for d in demos if d.instance_id == iid), demos[0]))


def _rollout(demo, resolved, parent=None, idx=0, traj=None, patch=None):
    r = copy.deepcopy(demo)
    r.instance_id = f"{demo.instance_id}#s{idx}"
    if traj is not None:
        r.trajectory = traj
    if patch is not None:
        r.patch = patch
    r.metadata = {"source": "student:test", "resolved": resolved, "parent_id": parent or demo.instance_id,
                  "sample_idx": idx}
    return r


def _bad_traj(demo, last_test_ok=True):
    """Edits the wrong thing, runs tests (own tests pass or fail), submits."""
    obs = "2 passed in 0.01s\n\n[exit code 0]" if last_test_ok else "1 failed, 1 passed in 0.01s\n\n[exit code 1]"
    return [
        Step("look", Action("view_file", {"path": "webapp/pagination.py"}), "..."),
        Step("edit", Action("str_replace", {"path": "webapp/pagination.py", "old_str": "x", "new_str": "y"}),
             "Error: old_str not found in webapp/pagination.py."),
        Step("test", Action("run_tests", {}), obs),
        Step("done", Action("submit", {}), "Submitted."),
    ]


BAD_PATCH = ("diff --git a/webapp/pagination.py b/webapp/pagination.py\n--- a/webapp/pagination.py\n"
             "+++ b/webapp/pagination.py\n@@ -5 +5 @@\n-    start = page * per_page\n+    start = page * per_page + 1\n"
             "diff --git a/conftest.py b/conftest.py\n--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1 @@\n+x = 1\n")


def test_features_on_engineer_demo():
    f = V.extract_features(_demo())
    assert f["ran_tests_before_submit"] == 1 and f["last_test_exit_zero"] == 1
    assert f["last_test_passed"] == 2 and f["last_test_failed"] == 0
    assert f["reproduced_issue"] == 1 and f["submitted"] == 1
    assert f["patch_files"] == 1 and f["patch_hunks"] == 1 and f["patch_lines"] == 2
    assert f["num_tool_errors"] == 0 and f["edited_test_infra"] == 0
    assert f["num_steps"] == len(_demo().trajectory)


def test_features_on_bad_rollout_and_hard_negative_tagging():
    d = _demo()
    hard = _rollout(d, False, idx=1, traj=_bad_traj(d, True), patch=BAD_PATCH)
    easy = _rollout(d, False, idx=2, traj=_bad_traj(d, False), patch=BAD_PATCH)
    good = _rollout(d, True, idx=3)
    f = V.extract_features(hard)
    assert f["num_tool_errors"] == 1 and f["reproduced_issue"] == 0 and f["edited_test_infra"] == 1
    assert f["patch_files"] == 2 and f["patch_hunks"] == 2
    fe = V.extract_features(easy)
    assert fe["last_test_exit_nonzero"] == 1 and fe["last_test_failed"] == 1 and fe["last_test_passed"] == 1
    assert V.tag_hard_negatives([hard, easy, good]) == 1
    assert hard.metadata["hard_negative"] is True
    assert easy.metadata["hard_negative"] is False and good.metadata["hard_negative"] is False


def test_parsers():
    assert V.parse_exit_code("a\n[exit code 1]\nb\n[exit code 0]") == 0
    assert V.parse_exit_code("nothing") is None
    assert V.parse_pytest_counts("3 failed, 7 passed, 1 error in 2s") == (7, 4)
    assert V.touches_test_infra(["src/a.py", "pkg/conftest.py", ".github/workflows/ci.yml", "setup.cfg"]) == [
        "pkg/conftest.py", ".github/workflows/ci.yml", "setup.cfg"]


def test_auroc_known_values():
    assert V.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert V.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert V.auroc([1, 2, 3], [0, 0, 1]) == 1.0
    assert V.auroc([3, 2, 1], [0, 0, 1]) == 0.0
    import math
    assert math.isnan(V.auroc([1, 2], [1, 1]))


def test_logreg_separable_and_roundtrip(tmp_path):
    rng = random.Random(0)
    X, y = [], []
    for _ in range(200):
        lab = rng.random() < 0.5
        X.append([rng.gauss(2 if lab else -2, 1), rng.gauss(0, 1), 5.0])  # last col constant (std=0)
        y.append(int(lab))
    v = V.FeatureVerifier(["a", "b", "c"]).fit_arrays(X, y, epochs=300)
    scores = [v.predict_proba_features(x) for x in X]
    assert V.auroc(scores, y) > 0.9
    p = tmp_path / "m.json"
    v.save(p)
    w = V.FeatureVerifier.load(p)
    assert [w.predict_proba_features(x) for x in X[:10]] == pytest.approx(scores[:10])
    assert w.predict_proba_features({"a": 3.0, "b": 0, "c": 5}) > 0.9


def test_fit_on_records_ranks_resolved_higher():
    d = _demo()
    recs = [_rollout(d, True, idx=i) for i in range(4)] + [
        _rollout(d, False, idx=10 + i, traj=_bad_traj(d, i % 2 == 0), patch=BAD_PATCH) for i in range(4)]
    v = V.FeatureVerifier().fit(recs)
    assert V.auroc([v(r) for r in recs], [V.label_of(r) for r in recs]) == 1.0


def test_judge_parsing_and_scoring():
    assert V.parse_judge_answer("blah\nVERDICT: YES\nCONFIDENCE: 0.8") == pytest.approx(0.8)
    assert V.parse_judge_answer("VERDICT: NO\nCONFIDENCE: 90%") == pytest.approx(0.1)
    assert V.parse_judge_answer("**VERDICT:** yes") == pytest.approx(0.85)
    assert V.parse_judge_answer("I think yes.") == pytest.approx(0.85)
    assert V.parse_judge_answer("unsure") is None
    seen = {}

    def llm(system, prompt):
        seen["prompt"] = prompt
        return "Looks right.\nVERDICT: YES\nCONFIDENCE: 0.7"

    j = V.JudgeVerifier(llm)
    d = _demo()
    assert j(d) == pytest.approx(0.7)
    assert "Pagination skips" in seen["prompt"] and "(page - 1)" in seen["prompt"] and "2 passed" in seen["prompt"]
    assert V.JudgeVerifier(lambda s, p: "???")(d) == 0.5
    assert V.JudgeVerifier(lambda s, p: 1 / 0)(d) == 0.5
    empty = copy.deepcopy(d)
    empty.patch = ""
    assert j(empty) == 0.0


def test_best_of_n_voting_and_tiebreak():
    d = _demo()
    alt = d.patch.replace("(page - 1) * per_page", "(page  -  1) * per_page")  # whitespace-only difference
    a = _rollout(d, True, idx=0)
    b = _rollout(d, True, idx=1, patch=alt)
    c = _rollout(d, False, idx=2, traj=_bad_traj(d, True), patch=BAD_PATCH)
    ranked = V.best_of_n([c, a, b], lambda r: 0.5)
    assert {ranked[0].index, ranked[1].index} == {1, 2} and ranked[2].index == 0
    assert ranked[0].agreement == pytest.approx(2 / 3)
    assert "agreement=2/3" in ranked[0].explanation
    # verifier can override voting when weight is low
    ranked = V.best_of_n([c, a, b], lambda r: 1.0 if r is c else 0.0, vote_weight=0.3)
    assert ranked[0].index == 0
    # tie-break on own tests passing
    x = _rollout(d, False, idx=3, traj=_bad_traj(d, False), patch=BAD_PATCH)
    y = _rollout(d, True, idx=4)
    ranked = V.best_of_n([x, y], lambda r: 0.5)
    assert ranked[0].index == 1 and ranked[0].own_tests_passed and not ranked[1].own_tests_passed


def test_evaluate_best_of_n():
    d = _demo()
    g1 = [_rollout(d, True, "p1", 0), _rollout(d, False, "p1", 1, _bad_traj(d, False), BAD_PATCH)]
    g2 = [_rollout(d, False, "p2", 0, _bad_traj(d, False), BAD_PATCH)] * 2
    res = V.evaluate_best_of_n(V.group_by_parent(g1 + g2), lambda r: float(r.metadata["resolved"]))
    assert res["n_instances"] == 2 and res["pass@1"] == pytest.approx(0.25)
    assert res["best_of_n"] == pytest.approx(0.5) and res["oracle_pass@n"] == pytest.approx(0.5)
    assert res["gap_closed"] == pytest.approx(1.0)


def _good_script(demo):
    return [(s.thought, s.action) for s in demo.trajectory]


def _broken_script():
    return [("wrong fix", Action("str_replace", {"path": "webapp/pagination.py",
                                                  "old_str": "start = page * per_page",
                                                  "new_str": "start = page * per_page + 1"})),
            ("done", Action("submit", {}))]


def _alternating_factory(demo, pattern):
    calls = {"i": 0}

    def factory():
        good = pattern[calls["i"] % len(pattern)]
        calls["i"] += 1
        return ScriptedPolicy(_good_script(demo) if good else _broken_script())
    return factory


def test_run_best_of_n_with_scripted_policies():
    d = _demo()
    repo = ROOT / d.repo
    res = V.run_best_of_n(_alternating_factory(d, [True, False, True]), d.problem_statement, repo, 3,
                          lambda r: 0.5, backend="local")
    assert len(res.candidates) == 3 and len(res.ranked) == 3
    assert V.normalize_patch(res.patch) == V.normalize_patch(d.patch)
    assert res.ranked[-1].index == 1  # the lone broken sample loses the vote


def _write_train_set(tmp_path):
    d = _demo()
    recs = []
    for p in range(6):
        recs += [_rollout(d, True, f"inst-{p}", 0), _rollout(d, True, f"inst-{p}", 1),
                 _rollout(d, False, f"inst-{p}", 2, _bad_traj(d, p % 2 == 0), BAD_PATCH)]
    path = tmp_path / "rollouts.jsonl"
    save_demos(recs, path)
    return path


def _parser():
    p = argparse.ArgumentParser()
    V.register(p.add_subparsers(dest="cmd", required=True))
    return p


def test_cli_verifier_train(tmp_path, capsys):
    path = _write_train_set(tmp_path)
    demos = tmp_path / "demos.jsonl"
    save_demos([_demo()], demos)
    out = tmp_path / "model.json"
    a = _parser().parse_args(["verifier-train", "--rollouts", str(path), "--demos", str(demos), "--out", str(out),
                              "--epochs", "200"])
    assert a.fn(a) == 0
    text = capsys.readouterr().out
    assert "dev AUROC" in text and "hard negatives" in text
    m = V.FeatureVerifier.load(out)
    assert m.meta["n_hard_negatives"] == 3

    excl = tmp_path / "frozen.json"
    excl.write_text(json.dumps({"instances": {"inst-3": "hash"}}))
    a = _parser().parse_args(["verifier-train", "--rollouts", str(path), "--out", str(out),
                              "--exclude-ids", str(excl)])
    assert a.fn(a) == 2


def test_exclude_guard_and_manifest_formats(tmp_path):
    (tmp_path / "a.txt").write_text("x\n\ny\n")
    (tmp_path / "b.json").write_text(json.dumps({"instances": ["x", "z"]}))
    assert V.load_exclude_ids(tmp_path / "a.txt") == {"x", "y"}
    assert V.load_exclude_ids(tmp_path / "b.json") == {"x", "z"}
    d = _demo()
    with pytest.raises(V.ExcludedInstanceError):
        V.check_not_excluded([_rollout(d, True, "x")], {"x"})
    V.check_not_excluded([_rollout(d, True, "ok")], {"x"})


def test_cli_best_of_n(tmp_path, monkeypatch, capsys):
    d = _demo()
    model = tmp_path / "model.json"
    V.FeatureVerifier().fit([_rollout(d, True, idx=0), _rollout(d, False, idx=1, traj=_bad_traj(d, False),
                                                                patch=BAD_PATCH)], epochs=50).save(model)
    factory = _alternating_factory(d, [True, False])
    monkeypatch.setattr(policies, "make_policy", lambda m: factory())
    issue = tmp_path / "issue.md"
    issue.write_text(d.problem_statement)
    report = tmp_path / "report.json"
    a = _parser().parse_args(["best-of-n", "--repo", str(ROOT / d.repo), "--issue-file", str(issue),
                              "--model", "student", "--n", "2", "--verifier", str(model), "--backend", "local",
                              "--out", str(report)])
    assert a.fn(a) == 0
    out = capsys.readouterr().out
    assert "(page - 1) * per_page" in out
    assert len(json.loads(report.read_text())["ranking"]) == 2
