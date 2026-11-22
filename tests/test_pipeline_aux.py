"""§3d auxiliary tasks: localization, patch-gen (str_replace round trip), test writing, repro, critique."""
import argparse
import copy
import json

import pytest

from repo_fix_agent import sandbox as sb
from repo_fix_agent.pipeline import aux_tasks as A
from repo_fix_agent.pipeline.sft_data import read_jsonl
from repo_fix_agent.schema import save_demos

try:
    from test_pipeline_sft import PATCH, make_demo
except ImportError:  # tests/ became a package
    from tests.test_pipeline_sft import PATCH, make_demo


@pytest.fixture(autouse=True)
def _local_sandbox(monkeypatch):
    monkeypatch.setattr(sb, "_docker_available", lambda: False)
    # once-per-process warning flags: restore them so test_sandbox's log assertions stay order-independent
    monkeypatch.setattr(sb, "_warned_fallback", sb._warned_fallback)
    monkeypatch.setattr(sb, "_warned_network", sb._warned_network)


@pytest.fixture
def demo(tmp_path):
    return make_demo(tmp_path)


def failed_rollout(d, k=0, patch=None):
    r = copy.deepcopy(d)
    r.instance_id = f"{d.instance_id}#s{k}"
    r.patch = patch if patch is not None else PATCH.replace("(page - 1) * per_page", "page * per_page - 1")
    r.metadata = {"source": "student:s", "resolved": False, "parent_id": d.instance_id}
    return r


# ---- parsing helpers --------------------------------------------------------------------------
def test_enclosing_symbols_innermost():
    src = "x = 1\n\nclass A:\n    def m(self):\n        return 1\n\n    y = 2\n\ndef f():\n    pass\n"
    names = [n for n, _, _ in A.enclosing_symbols(src, [1, 5, 7, 10])]
    assert names == ["A.m", "A", "f"]


def test_patch_parse_and_apply(demo, tmp_path):
    (fd,) = A.parse_patch(PATCH)
    assert fd.path == "webapp/pagination.py" and A.changed_old_lines(fd) == [5]
    orig = (tmp_path / "repo/webapp/pagination.py").read_text()
    assert "start = (page - 1) * per_page" in A.apply_file_diff(orig, fd)


# ---- localization -----------------------------------------------------------------------------
def test_localization_example_and_recall(demo, tmp_path):
    with A.RepoReader(demo, tmp_path) as r:
        ex = A.localization_example(demo, r)
    assert ex["task"] == "localization" and [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    user = ex["messages"][1]["content"]
    assert "webapp/pagination.py" in user and "tests/" not in user
    files, funcs = A.parse_localization(ex["messages"][2]["content"])
    assert files == ["webapp/pagination.py"] and funcs == ["webapp/pagination.py::paginate"]
    assert A.score_localization(["webapp/views.py", "webapp/pagination.py"], files, k=1) == 0.0
    assert A.score_localization(["webapp/views.py", "webapp/pagination.py"], files, k=2) == 1.0
    assert A.score_localization([], ["a", "b"], k=3) == 0.0


# ---- patch generation -------------------------------------------------------------------------
def test_str_replace_derivation_round_trips(demo, tmp_path):
    with A.RepoReader(demo, tmp_path) as r:
        ex = A.patchgen_example(demo, r)
    calls = A.parse_calls(ex["messages"][2]["content"])
    assert calls and all(c["tool"] == "str_replace" for c in calls)
    assert "def paginate" in ex["messages"][1]["content"]
    assert A.calls_to_patch(demo, calls, tmp_path).count("+    start = (page - 1) * per_page") == 1
    assert A.verify_patchgen(demo, ex["messages"][2]["content"], tmp_path).resolved
    bad = json.dumps([{"tool": "str_replace", "args": {"path": "webapp/pagination.py",
                                                         "old_str": "nope", "new_str": "x"}}])
    assert not A.verify_patchgen(demo, bad, tmp_path).resolved


def test_str_replace_handles_insertions_and_duplicates():
    orig = "a = 1\nb = 2\na = 1\nb = 2\nc = 3\n"
    new = "a = 1\nb = 2\na = 1\nb = 20\nINSERTED\nc = 3\n"
    calls = A._str_replace_calls("f.py", orig, new)
    text = orig
    for c in calls:
        assert text.count(c["args"]["old_str"]) == 1
        text = text.replace(c["args"]["old_str"], c["args"]["new_str"], 1)
    assert text == new


# ---- test writing / repro ---------------------------------------------------------------------
def test_verify_test_both_ways(demo, tmp_path):
    with A.RepoReader(demo, tmp_path) as r:
        ex = A.testwrite_example(demo, r, tmp_path, verify=True)
    assert ex is not None and ex["messages"][2]["content"].startswith("# file: tests/test_issue_001.py")
    trivial = "def test_ok():\n    assert True\n"
    v = A.verify_test(demo, trivial, tmp_path)
    assert v.before_rc == 0 and not v.ok  # passes on buggy code -> rejected
    broken = "from webapp.pagination import paginate\n\ndef test_x():\n    assert paginate([1], 1, 1) == [2]\n"
    assert not A.verify_test(demo, broken, tmp_path).ok  # fails both ways


def test_repro_script_verifies(demo, tmp_path):
    with A.RepoReader(demo, tmp_path) as r:
        ex = A.repro_example(demo, r, tmp_path)
    assert ex is not None and ex["meta"]["verified"]
    script = ex["messages"][2]["content"]
    v = A.verify_repro(demo, script, tmp_path)
    assert v.before_rc != 0 and v.after_rc == 0
    assert ex["messages"][1]["content"].startswith("<issue>")


# ---- critique ---------------------------------------------------------------------------------
def test_critique_labels(demo, tmp_path):
    rollouts = [failed_rollout(demo, 0), failed_rollout(demo, 1, patch=""),
                failed_rollout(demo, 2, patch=PATCH)]  # identical to gold -> skipped
    exs = A.critique_examples(demo, rollouts, tmp_path)
    labels = [e["meta"]["label"] for e in exs]
    assert labels == [True, False, False]
    assert exs[0]["messages"][2]["content"].startswith("RESOLVED")
    assert exs[1]["messages"][2]["content"].startswith("UNRESOLVED")
    assert "no changes" in exs[2]["messages"][2]["content"]
    assert A.score_critique("UNRESOLVED: wrong", exs[1]) and not A.score_critique("RESOLVED", exs[1])
    regraded = A.critique_examples(demo, rollouts[:1], tmp_path, regrade=True)
    assert "still wrong" in regraded[1]["messages"][2]["content"]


# ---- driver / CLI -----------------------------------------------------------------------------
def test_build_and_exclude(demo, tmp_path):
    exs = A.build_aux_tasks([demo], ["localization", "critique"], [failed_rollout(demo)], tmp_path)
    assert {e["task"] for e in exs} == {"localization", "critique"}
    assert len({e["id"] for e in exs}) == len(exs)
    with pytest.raises(ValueError, match="excluded"):
        A.build_aux_tasks([demo], ["localization"], root=tmp_path, exclude_ids={demo.instance_id})
    with pytest.raises(ValueError, match="excluded"):
        A.build_aux_tasks([], ["critique"], [failed_rollout(demo)], tmp_path, exclude_ids={demo.instance_id})
    with pytest.raises(ValueError, match="unknown"):
        A.build_aux_tasks([demo], ["nope"], root=tmp_path)


def test_cli_aux_tasks(demo, tmp_path, capsys):
    inp, rol, out = tmp_path / "d.jsonl", tmp_path / "r.jsonl", tmp_path / "aux.jsonl"
    save_demos([demo], inp)
    save_demos([failed_rollout(demo)], rol)
    p = argparse.ArgumentParser()
    A.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["aux-tasks", str(inp), "--out", str(out), "--rollouts", str(rol), "--root", str(tmp_path)])
    assert a.fn(a) == 0
    counts = json.loads(capsys.readouterr().out)["by_task"]
    assert counts == {"localization": 1, "patchgen": 1, "testwrite": 1, "repro": 1, "critique": 2}
    assert len(read_jsonl(out)) == 6
