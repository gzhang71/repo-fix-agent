"""Offline tests for pipeline/augment.py (§3h). Uses a tiny self-built seed instance."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from repo_fix_agent.grader import validate_instance
from repo_fix_agent.pipeline import augment as A
from repo_fix_agent.schema import Demo, load_demos, save_demos

TEST_CMD = "python -m pytest -q -p no:cacheprovider"


@pytest.fixture(autouse=True, scope="module")
def _restore_sandbox_warn_once():
    """grade() goes through make_sandbox("auto"), which flips sandbox's warn-once globals; restore
    them so tests elsewhere that assert on those warnings are not order-dependent."""
    from repo_fix_agent import sandbox as sb
    saved = sb._warned_fallback, sb._warned_network
    yield
    sb._warned_fallback, sb._warned_network = saved

BUGGY_CORE = textwrap.dedent('''\
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


    def total(items):
        return sum(items) + 1


    def unused_helper():
        return 42
''')
FIXED_CORE = BUGGY_CORE.replace("return sum(items) + 1", "return sum(items)")
EXISTING_TESTS = textwrap.dedent('''\
    from pkg.core import clamp


    def test_clamp_low():
        assert clamp(-5, 0, 10) == 0


    def test_clamp_high():
        assert clamp(50, 0, 10) == 10


    def test_clamp_mid():
        assert clamp(5, 0, 10) == 5
''')
HIDDEN_TESTS = textwrap.dedent('''\
    from pkg.core import total


    def test_total_sums():
        assert total([1, 2, 3]) == 6
''')


def _make_seed(root: Path, name: str = "seed-900-total") -> Demo:
    repo = root / "repos" / name
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "core.py").write_text(BUGGY_CORE)
    (repo / "tests" / "test_core.py").write_text(EXISTING_TESTS)
    return Demo(
        instance_id=name, repo=f"repos/{name}", base_commit=None,
        problem_statement="total() is off by one\n", test_cmd=TEST_CMD,
        test_patch=A._diff({}, {"tests/test_hidden.py": HIDDEN_TESTS}),
        FAIL_TO_PASS=["tests/test_hidden.py::test_total_sums"],
        PASS_TO_PASS=["tests/test_core.py::test_clamp_low", "tests/test_core.py::test_clamp_high",
                      "tests/test_core.py::test_clamp_mid"],
        patch=A._diff({"pkg/core.py": BUGGY_CORE}, {"pkg/core.py": FIXED_CORE}),
        metadata={"source": "engineer_seed", "language": "python"},
    )


@pytest.fixture(scope="module")
def seed_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("aug")
    demo = _make_seed(root)
    assert validate_instance(demo, root) == []
    stats = A.AugmentStats()
    new = A.mutate_demo(demo, root, root / "synth", max_mutants=2, seed=0, max_tries=12, stats=stats)
    return root, demo, new, stats


# ---- operators -------------------------------------------------------------------------------
OP_CASES = [
    ("compare_flip", "def f(a, b):\n    return a < b\n", "    return a <= b"),
    ("compare_flip", "def f(a, b):\n    return a != b\n", "    return a == b"),
    ("const_off_by_one", "X = 10\n", "X = 11"),
    ("slice_bound", "def f(xs, n):\n    return xs[:n]\n", "    return xs[:n + 1]"),
    ("bool_op_swap", "def f(a, b):\n    return a and b\n", "    return a or b"),
    ("drop_not", "def f(a):\n    return not a\n", "    return a"),
    ("arith_swap", "def f(a, b):\n    return a + b\n", "    return a - b"),
    ("arith_swap", "def f(a, b):\n    return a / b\n", "    return a // b"),
    ("return_none", "def f(a):\n    if a:\n        return a * 2\n    return 0\n", "        return None"),
    ("remove_call", "def f(s):\n    return s.strip()\n", "    return s"),
    ("range_minus_one", "def f(n):\n    return list(range(n))\n", "    return list(range(n - 1))"),
    ("get_to_index", "def f(d):\n    return d.get('k', 0)\n", "    return d['k']"),
]


@pytest.mark.parametrize("op,src,expected", OP_CASES)
def test_operator_single_line_change(op, src, expected):
    muts = A.enumerate_mutations(src, "m.py", [op])
    assert muts, f"no {op} mutation found"
    outs = [A.apply_mutation(src, m) for m in muts]
    for new in outs:
        ast.parse(new)
        changed = [(a, b) for a, b in zip(src.splitlines(), new.splitlines()) if a != b]
        assert len(changed) == 1 and len(src.splitlines()) == len(new.splitlines())
    assert any(expected in new.splitlines() for new in outs), outs


def test_operators_skip_string_concat_and_repetition():
    src = "def f(a):\n    return 'x' + a, [0] * a\n"
    assert A.enumerate_mutations(src, "m.py", ["arith_swap"]) == []


def test_enumeration_is_deterministic():
    assert A.enumerate_mutations(BUGGY_CORE, "c.py") == A.enumerate_mutations(BUGGY_CORE, "c.py")


# ---- bug injection ---------------------------------------------------------------------------
def test_mutants_are_valid_instances(seed_env):
    root, demo, new, stats = seed_env
    assert len(new) >= 1, stats.to_dict()
    for d in new:
        assert d.instance_id.startswith(f"{demo.instance_id}-mut")
        assert d.trajectory == []
        md = d.metadata
        assert md["source"] == "synthetic:mutation" and md["parent_id"] == demo.instance_id
        assert set(md["mutation"]) == {"operator", "file", "line", "before", "after"}
        assert md["bug_type"] == A.BUG_TYPES[md["mutation"]["operator"]]
        assert md["difficulty"] in {"easy", "medium", "hard"}
    assert validate_instance(new[0], root) == []


def test_f2p_hidden_from_repo_and_in_test_patch(seed_env):
    root, _, new, _ = seed_env
    for d in new:
        repo = root / d.repo
        for t in d.FAIL_TO_PASS:
            f = t.split("::")[0]
            assert not (repo / f).exists(), f"{f} visible to the agent"
            assert f"+++ b/{f}" in d.test_patch
            assert t.split("::")[1] in d.test_patch
        assert not set(d.FAIL_TO_PASS) & set(d.PASS_TO_PASS)


def test_patch_is_inverse_of_mutation(seed_env, tmp_path):
    root, demo, new, _ = seed_env
    d = new[0]
    mut = d.metadata["mutation"]
    assert d.metadata["files_changed"] == [mut["file"]]
    minus = [l[1:] for l in d.patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    plus = [l[1:] for l in d.patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert minus == [mut["after"]] and plus == [mut["before"]]
    work = tmp_path / "w"
    A._copy_tree(root / d.repo, work)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    A._apply(work, d.patch)
    good = A.known_good(demo, root, tmp_path / "good")
    assert (work / mut["file"]).read_text() == (good / mut["file"]).read_text()


def test_issue_text_has_no_test_names(seed_env):
    _, _, new, _ = seed_env
    for d in new:
        assert "test_" not in d.problem_statement
        assert d.metadata["mutation"]["after"].strip() not in d.problem_statement
        assert ".py" not in d.problem_statement


def test_mutants_breaking_no_test_are_dropped(tmp_path):
    # unused_helper() is not covered by any test, so mutating its constant breaks nothing
    demo = _make_seed(tmp_path)
    good = A.known_good(demo, tmp_path, tmp_path / "good")
    m = next(m for m in A.enumerate_mutations(FIXED_CORE, "pkg/core.py", ["const_off_by_one"])
             if m.replacement == "43")
    (good / "pkg/core.py").write_text(A.apply_mutation(FIXED_CORE, m))
    res = A.run_suite(good)
    assert res.returncode == 0 and not res.failed
    # end to end: remove_call only matches the uncovered helper -> every mutant is dropped
    helper_old, helper_new = "def unused_helper():\n    return 42", "def unused_helper(s):\n    return s.strip()"
    (tmp_path / demo.repo / "pkg/core.py").write_text(BUGGY_CORE.replace(helper_old, helper_new))
    demo.patch = A._diff({"pkg/core.py": BUGGY_CORE.replace(helper_old, helper_new)},
                         {"pkg/core.py": FIXED_CORE.replace(helper_old, helper_new)})
    stats = A.AugmentStats()
    out = A.mutate_demo(demo, tmp_path, tmp_path / "synth", operators=["remove_call"], stats=stats)
    assert out == [] and stats.dropped["no_failing_tests"] >= 1


def test_determinism_given_seed(seed_env, tmp_path):
    root, demo, new, _ = seed_env
    again = A.mutate_demo(demo, root, tmp_path / "synth2", max_mutants=2, seed=0, max_tries=12)
    key = lambda ds: [(d.instance_id, d.metadata["mutation"], d.FAIL_TO_PASS, d.patch) for d in ds]
    assert key(again) == key(new)


def test_exclude_ids_refusal(tmp_path):
    demo = _make_seed(tmp_path)
    with pytest.raises(ValueError, match="excluded"):
        A.augment([demo], tmp_path, tmp_path / "synth", exclude_ids={demo.instance_id})
    with pytest.raises(ValueError):
        A.perturb_repo(demo, tmp_path / "p", root=tmp_path, exclude_ids={demo.instance_id})
    child = Demo(**{**demo.__dict__, "instance_id": "x-reph-terse", "metadata": {"parent_id": demo.instance_id}})
    with pytest.raises(ValueError):
        A.rephrase_issue(child, lambda s, p: "fine", "terse", exclude_ids={demo.instance_id})


def test_load_exclude_ids_formats(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"instances": {"a": "hash1", "b": "hash2"}}))
    assert A.load_exclude_ids(p) == {"a", "b"}
    p.write_text(json.dumps({"instances": [{"instance_id": "c"}, "d"]}))
    assert A.load_exclude_ids(p) == {"c", "d"}
    p.write_text("e\nf\n\n")
    assert A.load_exclude_ids(p) == {"e", "f"}


# ---- issue writing ---------------------------------------------------------------------------
FAIL_OUT = textwrap.dedent('''\
    =================================== FAILURES ===================================
    _______________________________ test_total_sums ________________________________
    tests/test_hidden.py:5: in test_total_sums
        assert total([1, 2, 3]) == 6
    E   assert 7 == 6
    E    +  where 7 = total([1, 2, 3])
    =========================== short test summary info ============================
    FAILED tests/test_hidden.py::test_total_sums - assert 7 == 6
''')


def test_fallback_issue_template():
    text = A.write_issue(FAIL_OUT, "def total(items):\n    return sum(items) + 1\n")
    assert "test_" not in text and "tests/" not in text and ".py" not in text
    assert "Calling `total([1, 2, 3])` returns `7` but `6` was expected." in text


def test_fallback_issue_exception():
    out = textwrap.dedent('''\
        ______ test_lookup ______
        tests/test_x.py:4: in test_lookup
            assert lookup({}, "a") is None
        pkg/core.py:2: in lookup
            return d["a"]
        E   KeyError: 'a'
    ''')
    text = A.write_issue(out, "")
    assert "KeyError" in text and "test_" not in text


def test_llm_issue_writer_rejects_leak_and_retries():
    calls = []

    def fake(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "test_total_sums fails\n\nThe test test_total_sums is red."
        return "total() overcounts\n\n`total([1, 2, 3])` gives 7 instead of 6."

    text = A.write_issue(FAIL_OUT, "def total(items): ...", fake, forbidden=["test_total_sums"])
    assert len(calls) == 2 and "rejected" in calls[1]
    assert text.startswith("total() overcounts")


def test_llm_issue_writer_falls_back_after_two_leaks():
    fake = lambda s, p: "bug: return sum(items) + 1 is wrong"
    text = A.write_issue(FAIL_OUT, "", fake, forbidden=["return sum(items) + 1"])
    assert text.startswith("`total` returns wrong result")


# ---- rephrase / perturb ----------------------------------------------------------------------
def test_rephrase_with_fake_llm(tmp_path):
    demo = _make_seed(tmp_path)
    seen = {}

    def fake(system, prompt):
        seen["prompt"] = prompt
        return "total() adds one too many"

    r = A.rephrase_issue(demo, fake, "terse")
    assert r.instance_id == f"{demo.instance_id}-reph-terse"
    assert r.metadata["source"] == "synthetic:rephrase" and r.metadata["style"] == "terse"
    assert r.metadata["parent_id"] == demo.instance_id
    assert (r.repo, r.patch, r.test_patch, r.FAIL_TO_PASS) == (demo.repo, demo.patch, demo.test_patch, demo.FAIL_TO_PASS)
    assert "terse" in seen["prompt"] and demo.problem_statement.strip() in seen["prompt"]
    with pytest.raises(ValueError):
        A.rephrase_issue(demo, lambda s, p: "see test_total_sums", "user-perspective")
    with pytest.raises(ValueError):
        A.rephrase_issue(demo, fake, "poetic")


def test_perturb_keeps_validation(tmp_path):
    demo = _make_seed(tmp_path)
    # an unreferenced module that may be renamed
    (tmp_path / demo.repo / "pkg" / "orphan_mod.py").write_text("VALUE = 1\n")
    p = A.perturb_repo(demo, tmp_path / "synth", seed=3, root=tmp_path)
    assert p is not None
    assert p.metadata["source"] == "synthetic:perturb" and p.metadata["parent_id"] == demo.instance_id
    added = p.metadata["perturbation"]["added"]
    assert 1 <= len(added) <= 3
    repo = tmp_path / p.repo
    for rel in added:
        assert (repo / rel).exists()
        ast.parse((repo / rel).read_text())
    ren = p.metadata["perturbation"]["renamed"]
    if ren:
        assert not (repo / ren[0]).exists() and (repo / ren[1]).exists()
    assert validate_instance(p, tmp_path) == []


# ---- driver / CLI ----------------------------------------------------------------------------
def test_cli_end_to_end(tmp_path, capsys):
    root = tmp_path
    demo = _make_seed(root)
    save_demos([demo], root / "in.jsonl")
    (root / "frozen.txt").write_text("some-other-id\n")
    p = argparse.ArgumentParser(prog="repo-fix")
    A.register(p.add_subparsers(dest="cmd"))
    out = root / "out.jsonl"
    a = p.parse_args(["augment", str(root / "in.jsonl"), "--out", str(out), "--repos-dir", str(root / "synth"),
                      "--max-per-demo", "1", "--max-tries", "6", "--kinds", "mutation,rephrase",
                      "--root", str(root), "--exclude-ids", str(root / "frozen.txt")])
    assert a.fn(a) == 0
    got = load_demos(out)
    assert len(got) == 1 and got[0].metadata["source"] == "synthetic:mutation"
    assert not Path(got[0].repo).is_absolute()  # repos dir inside root -> relative path
    assert validate_instance(got[0], root) == []
    stats = json.loads((root / "out.jsonl.stats.json").read_text())
    assert stats["dropped"].get("rephrase_no_llm") == 1
    assert "--root" in capsys.readouterr().out
    # frozen-test guard through the CLI
    (root / "frozen.txt").write_text(demo.instance_id + "\n")
    assert a.fn(a) == 2


def test_fallback_never_quotes_source_frames():
    out = textwrap.dedent('''\
        ______ test_partial ______
        tests/test_cfg.py:6: in test_partial
            assert load_config(None)["retries"] == 3
        pkg/config.py:4: in load_config
            cfg.update(overrides and {})
        E   TypeError: 'NoneType' object is not iterable
    ''')
    text = A.write_issue(out, "def load_config(overrides=None):\n    ...", forbidden=["cfg.update(overrides and {})"],
                         known_funcs={"load_config"})
    assert "cfg.update" not in text and "test_" not in text
    assert "`load_config(None)`" in text and "TypeError" in text
