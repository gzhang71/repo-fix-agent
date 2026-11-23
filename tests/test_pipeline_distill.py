"""Offline tests for pipeline/distill.py (§3b hint ladder + leakage filter, §3c rejection sampling)."""
import argparse
import dataclasses
import json
from pathlib import Path

import pytest

from repo_fix_agent import grader, sandbox
from repo_fix_agent.pipeline import distill as D
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action, Demo, Step, load_demos, save_demos

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "demos" / "seed.jsonl"
PAG, LEAP = "seed-001-pagination-offset", "seed-003-leap-year-century"



@pytest.fixture(autouse=True)
def _no_submit_gate(monkeypatch):
    """These tests replay fixed gold scripts, which run the suite but no scratch_*.py repro, so the
    submit gate would bounce their submit. The gate itself is tested in test_student_improve.py."""
    from repo_fix_agent import agent
    monkeypatch.setattr(agent.SubmitGate, "intercept", lambda self, tool: None)

@pytest.fixture(autouse=True)
def _local_sandbox(monkeypatch):
    # grader.grade builds RepoEnv(backend="auto"); keep it local and skip the `docker info` probe.
    monkeypatch.setattr(sandbox, "_docker_available", lambda: False)
    # make_sandbox flips a module-level warn-once flag; restore it so other test files still see the warning
    monkeypatch.setattr(sandbox, "_warned_fallback", getattr(sandbox, "_warned_fallback", False), raising=False)


@pytest.fixture(scope="module")
def seeds():
    by_id = {d.instance_id: d for d in load_demos(SEED)}
    if PAG not in by_id or LEAP not in by_id:
        pytest.skip("seed instances not available")
    return by_id


_GRADE_CACHE = {}


def cached_grade(demo, patch, root):
    key = (demo.instance_id, patch)
    if key not in _GRADE_CACHE:
        _GRADE_CACHE[key] = grader.grade(demo, patch, root)
    return _GRADE_CACHE[key]


def script(steps):
    return [(s.thought, s.action) for s in steps]


def without(demo, pred):
    return [s for s in demo.trajectory if not pred(s)]


def is_tool(name):
    return lambda s: s.action.tool == name


def is_test_run(s):
    return s.action.tool == "bash" and "pytest" in s.action.args.get("command", "")


def run(demos, factory, **kw):
    kw.setdefault("k", 2)
    kw.setdefault("hints", False)
    return D.distill(demos, factory, ROOT, backend="local", grade_fn=cached_grade, **kw)


class RecordingPolicy(ScriptedPolicy):
    """Solves only when the system prompt contains `needle`; otherwise runs `fallback`."""
    created = []

    def __init__(self, solve, fallback, needle=None):
        super().__init__(solve)
        self.solve, self.fallback, self.needle = solve, fallback, needle
        self.system = None
        self.served_models = {"fake-teacher-1"}
        RecordingPolicy.created.append(self)

    def reset(self, system, task):
        self.system = system
        self._steps = self.solve if self.needle is None or self.needle in system else self.fallback
        super().reset(system, task)


# ---- rollouts ---------------------------------------------------------------------------------
def test_rollout_instance_resolves_and_records_metadata(seeds):
    demo = seeds[PAG]
    out = D.rollout_instance(lambda: RecordingPolicy(script(demo.trajectory), []), demo, ROOT, k=2,
                             model_name="claude-x", backend="local", grade_fn=cached_grade)
    assert len(out) == 2
    for i, r in enumerate(out):
        m = r.metadata
        assert r.instance_id == demo.instance_id and r.test_patch == demo.test_patch
        assert m["resolved"] and m["source"] == "teacher:claude-x" and m["parent_id"] == demo.instance_id
        assert m["sample_idx"] == i and m["hint_level"] == 0 and m["weight"] == 1.0
        assert m["ran_tests_before_submit"] and m["leakage_flags"] == []
        assert m["served_models"] == ["fake-teacher-1"]
        assert all(m["grade"]["fail_to_pass"].values())
        assert [s.action for s in r.trajectory] == [s.action for s in demo.trajectory]
        assert r.patch == demo.patch


def test_distill_splits_resolved_failed_and_dedups(seeds):
    pag, leap = seeds[PAG], seeds[LEAP]
    broken_leap = without(leap, is_tool("str_replace"))  # drop the fix step -> unresolved
    factories = {PAG: script(pag.trajectory), LEAP: script(broken_leap)}
    current = {}

    def factory():
        return ScriptedPolicy(factories[current["id"]])

    # one distill call per instance so each gets its own scripted factory
    kept, failed = [], []
    stats_all = []
    for d in (pag, leap):
        current["id"] = d.instance_id
        k_, f_, s_ = run([d], factory)
        kept += k_
        failed += f_
        stats_all.append(s_)
    assert [r.instance_id for r in kept] == [PAG]               # 2 identical resolved -> 1 kept
    assert stats_all[0]["dropped"]["duplicate"] == 1
    assert len(failed) == 2 and all(not r.metadata["resolved"] for r in failed)
    assert stats_all[1]["dropped"]["unresolved"] == 2
    assert stats_all[1]["resolve_rate_by_hint_level"]["0"]["rate"] == 0.0
    assert stats_all[0]["cost"]["steps"] == 2 * len(pag.trajectory)


def test_requires_tests_before_submit(seeds):
    pag = seeds[PAG]
    no_tests = without(pag, is_test_run)
    kept, failed, stats = run([pag], lambda: ScriptedPolicy(script(no_tests)), k=1)
    assert kept == [] and failed == []
    assert stats["dropped"]["no_tests_before_submit"] == 1
    assert stats["resolve_rate_by_hint_level"]["0"]["resolved"] == 1


def test_ran_tests_before_submit_variants():
    edit = Step("", Action("str_replace", {"path": "a.py", "old_str": "x", "new_str": "y"}), "Edited a.py.")
    sub = Step("", Action("submit", {}), "Submitted.")
    t = lambda cmd: Step("", Action("bash", {"command": cmd}), "ok")  # noqa: E731
    assert D.ran_tests_before_submit([edit, t("python -m pytest -q"), sub])
    assert D.ran_tests_before_submit([edit, t("python -m unittest discover"), sub])
    assert D.ran_tests_before_submit([edit, Step("", Action("run_tests", {}), "1 passed"), sub])
    assert not D.ran_tests_before_submit([t("pytest -q"), edit, sub])  # only before the edit
    assert not D.ran_tests_before_submit([edit, sub, t("pytest")])


# ---- hint ladder -------------------------------------------------------------------------------
def test_build_hint_levels(seeds):
    leap = seeds[LEAP]
    h = [D.build_hint(leap, lv) for lv in range(5)]
    assert h[0] == ""
    assert all(t in h[1] for t in leap.FAIL_TO_PASS) and "calendar_utils/dates.py" not in h[1]
    assert "Files changed by the fix" in h[2] and "calendar_utils/dates.py" in h[2]
    assert leap.test_patch.strip() in h[3] and leap.patch.strip() not in h[3]
    assert leap.patch.strip() in h[4] and "reference solution an engineer wrote" in h[4]
    assert "explore the repository normally" in h[4]
    with pytest.raises(ValueError):
        D.build_hint(leap, 5)


def test_hint_ladder_escalates_and_hint_not_saved(seeds):
    leap = seeds[LEAP]
    RecordingPolicy.created.clear()
    solve = script(leap.trajectory)
    fail = script(without(leap, is_tool("str_replace")))
    kept, failed, stats = run([leap], lambda: RecordingPolicy(solve, fail, needle="Files changed by the fix"),
                              k=2, hints=True)
    levels = [p.system for p in RecordingPolicy.created]
    assert len(levels) == 6                                # levels 0,1,2 x k=2; stops at level 2
    assert "<hint>" not in levels[0] and "<hint>" not in levels[1]
    assert D.build_hint(leap, 2) in levels[4] and "calendar_utils/dates.py" in levels[4]
    assert stats["resolve_rate_by_hint_level"]["2"]["rate"] == 1.0
    assert "3" not in stats["resolve_rate_by_hint_level"]
    assert stats["per_instance"][LEAP]["resolved_level"] == 2
    assert len(kept) == 1 and kept[0].metadata["hint_level"] == 2  # 2 identical -> deduped to 1
    assert kept[0].metadata["served_models"] == ["fake-teacher-1"]
    assert len(failed) == 4
    saved = json.dumps(kept[0].to_dict())
    assert D.HINT_HEADER not in saved and "<hint>" not in saved and "Files changed by the fix" not in saved


def test_no_hints_means_no_escalation(seeds):
    leap = seeds[LEAP]
    fail = script(without(leap, is_tool("str_replace")))
    kept, failed, stats = run([leap], lambda: ScriptedPolicy(fail), k=1, hints=False)
    assert kept == [] and len(failed) == 1 and list(stats["resolve_rate_by_hint_level"]) == ["0"]


def test_hinted_policy_delegates():
    inner = RecordingPolicy([("t", Action("submit", {}))], [])
    hp = D.HintedPolicy(inner, "HINT-TEXT", 1)
    hp.reset("SYS", "TASK")
    assert inner.system.startswith("SYS") and "HINT-TEXT" in inner.system
    assert hp.act(None) == ("t", Action("submit", {}))
    assert hp.served_models == {"fake-teacher-1"}
    hp.preload([])


# ---- leakage ------------------------------------------------------------------------------------
def leaky_direct_edit(demo):
    """First step is the edit, no exploration; then run tests and submit."""
    edit = next(s for s in demo.trajectory if s.action.tool == "str_replace")
    return [edit] + [s for s in demo.trajectory if is_test_run(s)] + [demo.trajectory[-1]]


def test_leaky_rollouts_dropped_when_hinted_flagged_when_unhinted(seeds):
    pag = seeds[PAG]
    leaky = script(leaky_direct_edit(pag))
    # unhinted: resolved + leaky -> kept but flagged
    kept, _, stats = run([pag], lambda: ScriptedPolicy(leaky), k=1)
    assert len(kept) == 1 and "leak_unsurfaced_first_edit" in kept[0].metadata["leakage_flags"]
    assert stats["flagged_unhinted"]["leak_unsurfaced_first_edit"] == 1
    # hinted: unhinted attempt fails, hinted attempt is leaky -> dropped (escalation stops: it resolved)
    fail = script(without(pag, is_tool("str_replace")))
    kept, failed, stats = run([pag], lambda: RecordingPolicy(leaky, fail, needle="<hint>"), k=1, hints=True)
    assert kept == [] and len(failed) == 1
    assert stats["dropped"]["leak_unsurfaced_first_edit"] == 1
    assert list(stats["resolve_rate_by_hint_level"]) == ["0", "1"]


def test_thought_mentioning_reference_is_dropped_when_hinted(seeds):
    pag = seeds[PAG]
    steps = [dataclasses.replace(s) for s in pag.trajectory]
    i = next(j for j, s in enumerate(steps) if s.action.tool == "str_replace")
    steps[i] = Step("Apply the fix from the reference solution.", steps[i].action, steps[i].observation)
    fail = script(without(pag, is_tool("str_replace")))
    kept, _, stats = run([pag], lambda: RecordingPolicy(script(steps), fail, needle="<hint>"), k=1, hints=True)
    assert kept == [] and stats["dropped"]["leak_thought_mentions"] == 1


def test_leakage_flag_heuristics(seeds):
    pag = seeds[PAG]
    assert D.leakage_flags(pag, pag.trajectory, pag.patch) == []
    # (b) no exploration before first edit and the patch touches >1 file
    two_file_patch = pag.patch + pag.patch.replace("webapp/pagination.py", "webapp/other.py")
    edit_first = leaky_direct_edit(pag)
    flags = D.leakage_flags(pag, edit_first, two_file_patch)
    assert "leak_no_exploration" in flags and "leak_unsurfaced_first_edit" in flags
    assert "leak_no_exploration" not in D.leakage_flags(pag, edit_first, pag.patch)
    # (c) test name mentioned before it was ever observed
    t0 = pag.trajectory[0]
    leaf = pag.FAIL_TO_PASS[0].split("::")[-1]
    named = [Step(f"Make {leaf} pass.", t0.action, t0.observation)] + pag.trajectory[1:]
    assert "leak_thought_mentions" in D.leakage_flags(pag, named, pag.patch)
    # (c) quoting an added gold line before viewing the file
    quoted = [Step("Should be `start = (page - 1) * per_page`.", t0.action, t0.observation)] + pag.trajectory[1:]
    assert "leak_thought_mentions" in D.leakage_flags(pag, quoted, pag.patch)
    # ... but the same quote after viewing the file is fine
    view_i = next(j for j, s in enumerate(pag.trajectory) if s.action.tool == "view_file")
    after = list(pag.trajectory)
    after[view_i + 1] = Step("Should be `start = (page - 1) * per_page`.", after[view_i + 1].action,
                             after[view_i + 1].observation)
    assert D.leakage_flags(pag, after, pag.patch) == []


def test_fast_first_edit_downweights_hinted(seeds):
    pag, leap = seeds[PAG], seeds[LEAP]
    grep = leap.trajectory[0]                      # surfaces calendar_utils/dates.py
    assert grep.action.tool == "bash"
    edit = next(s for s in leap.trajectory if s.action.tool == "str_replace")
    tail = [s for s in leap.trajectory if is_test_run(s)] + [leap.trajectory[-1]]
    fast = script([grep, Step("Fix the century rule.", edit.action, edit.observation)] + tail)
    fail = script(without(leap, is_tool("str_replace")))
    plans = {PAG: lambda: ScriptedPolicy(script(pag.trajectory)),
             LEAP: lambda: RecordingPolicy(fast, fail, needle="<hint>")}
    order = iter([PAG] * 3 + [LEAP] * 6)           # pag: 3 unhinted; leap: 3 unhinted + 3 level-1
    kept, _, stats = run([pag, leap], lambda: plans[next(order)](), k=3, hints=True)
    by = {r.instance_id: r for r in kept}
    assert by[PAG].metadata["weight"] == 1.0       # unhinted reference: steps-to-first-edit = 3
    assert by[LEAP].metadata["hint_level"] == 1 and by[LEAP].metadata["steps_to_first_edit"] == 1
    assert by[LEAP].metadata["weight"] == 0.5 and "fast_first_edit" in by[LEAP].metadata["leakage_flags"]
    assert stats["downweighted"] == 1


# ---- dedup + cap ---------------------------------------------------------------------------------
def test_dedup_and_cap_prefers_shorter(seeds):
    pag = seeds[PAG]
    i = next(j for j, s in enumerate(pag.trajectory) if s.action.tool == "str_replace")
    view = next(s for s in pag.trajectory if s.action.tool == "view_file")

    def variant(new_str, extra_views):
        steps = list(pag.trajectory)
        a = steps[i].action
        steps[i] = Step(steps[i].thought, Action("str_replace", {**a.args, "new_str": new_str}), "")
        return script(steps[:i] + [view] * extra_views + steps[i:])

    variants = iter([variant("start = per_page * (page - 1)", 2),
                     variant("start = (page - 1) * per_page", 0),
                     variant("start = (page-1) * per_page", 1)])
    kept, _, stats = run([pag], lambda: ScriptedPolicy(next(variants)), k=3, max_per_instance=2)
    assert len(kept) == 2 and stats["dropped"]["over_cap"] == 1
    assert sorted(len(r.trajectory) for r in kept) == [len(pag.trajectory), len(pag.trajectory) + 1]


def _rec(iid, n_steps, patch, idx, tool="view_file"):
    steps = [Step("", Action(tool, {"path": f"f{j}.py"}), "") for j in range(n_steps)]
    return Demo(iid, "r", None, "p", "pytest", "", [], [], patch, steps, {
        "parent_id": iid, "resolved": True, "ran_tests_before_submit": True, "hint_level": 0,
        "leakage_flags": [], "weight": 1.0, "sample_idx": idx, "steps_to_first_edit": None,
        "patch_bucket": "1-file-1-hunk"})


def test_filter_dedups_identical_normalized_patch():
    p1 = "diff --git a/x.py b/x.py\nindex 111..222 100644\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@ def f():\n-a\n+b\n"
    p2 = p1.replace("index 111..222", "index 333..444").replace("def f():", "")
    recs = [_rec("i", 5, p1, 0), _rec("i", 3, p2, 1, tool="search"), _rec("i", 4, p1 + "+c\n", 2)]
    kept, _, dropped = D.filter_rollouts(recs, max_per_instance=2)
    assert [r.metadata["sample_idx"] for r in kept] == [1, 2] and dropped["duplicate"] == 1


# ---- mixing --------------------------------------------------------------------------------------
def _pool(prefix, n_inst, per, source):
    out = []
    for i in range(n_inst):
        for j in range(per):
            d = _rec(f"{prefix}{i}", 3 + j, f"p{i}{j}", j)
            d.metadata["source"] = source
            out.append(d)
    return out


def test_mix_ratios_and_caps():
    teacher = _pool("inst", 10, 3, "teacher:m")
    engineer = _pool("inst", 10, 1, "engineer")
    for d in engineer:
        del d.metadata["parent_id"]
    m = D.mix_datasets(teacher, engineer, (3, 1), seed=0)
    src = [d.metadata["mix_source"] for d in m]
    assert src.count("teacher") == 30 and src.count("engineer") == 10
    m = D.mix_datasets(teacher, engineer, (1, 1), seed=0)
    t = [d for d in m if d.metadata["mix_source"] == "teacher"]
    assert len(t) == 10 and len({d.instance_id for d in t}) == 10   # spread across instances
    m = D.mix_datasets(teacher, engineer, (3, 1), seed=0, cap_per_instance=1)
    src = [d.metadata["mix_source"] for d in m]
    assert src.count("teacher") == 10 and src.count("engineer") == 3
    assert all(len(d.trajectory) == 3 for d in m if d.metadata["mix_source"] == "teacher")  # shortest kept
    assert len(D.mix_datasets(teacher, engineer, (1, 0))) == 30
    assert [d.instance_id for d in D.mix_datasets(teacher, engineer, (3, 1), seed=7)] == \
           [d.instance_id for d in D.mix_datasets(teacher, engineer, (3, 1), seed=7)]
    assert teacher[0].metadata.get("mix_source") is None  # inputs not mutated


# ---- frozen-test guard -----------------------------------------------------------------------------
def test_exclude_ids_refuses_before_running(seeds):
    pag = seeds[PAG]
    calls = []
    factory = lambda: calls.append(1) or ScriptedPolicy([])  # noqa: E731
    with pytest.raises(D.FrozenInstanceError):
        D.distill([pag], factory, ROOT, exclude_ids={PAG})
    with pytest.raises(D.FrozenInstanceError):
        D.rollout_instance(factory, pag, ROOT, exclude_ids={PAG})
    with pytest.raises(D.FrozenInstanceError):
        D.distill_instance(factory, pag, ROOT, exclude_ids={PAG})
    assert calls == []


def test_load_exclude_ids_formats(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"instances": {"a": "h1", "b": "h2"}}))
    assert D.load_exclude_ids(f) == {"a", "b"}
    f.write_text(json.dumps({"instances": ["a", {"instance_id": "c"}]}))
    assert D.load_exclude_ids(f) == {"a", "c"}
    f.write_text(json.dumps(["x"]))
    assert D.load_exclude_ids(f) == {"x"}
    t = tmp_path / "ids.txt"
    t.write_text("# frozen\na\n\nb\n")
    assert D.load_exclude_ids(t) == {"a", "b"}


# ---- CLI ---------------------------------------------------------------------------------------------
def _parser():
    p = argparse.ArgumentParser(prog="repo-fix")
    D.register(p.add_subparsers(dest="cmd", required=True))
    return p


def test_cli_distill_and_mix_end_to_end(seeds, tmp_path, monkeypatch):
    pag = seeds[PAG]
    models = []

    def fake_make_policy(model):
        models.append(model)
        return ScriptedPolicy(script(pag.trajectory))

    monkeypatch.setattr(D.policies, "make_policy", fake_make_policy)
    src = tmp_path / "in.jsonl"
    save_demos([pag], src)
    r, f, s = tmp_path / "r.jsonl", tmp_path / "f.jsonl", tmp_path / "s.json"
    a = _parser().parse_args(["distill", str(src), "--model", "claude-test", "--k", "2", "--out-resolved", str(r),
                              "--out-failed", str(f), "--stats", str(s), "--root", str(ROOT), "--hints",
                              "--backend", "local"])
    assert a.fn(a) == 0
    kept = load_demos(r)
    assert len(kept) == 1 and kept[0].metadata["source"] == "teacher:claude-test"
    assert load_demos(f) == [] and models == ["claude-test"] * 2
    stats = json.loads(s.read_text())
    assert stats["kept"] == 1 and stats["dropped"]["duplicate"] == 1 and stats["cost"]["episodes"] == 2

    ex = tmp_path / "frozen.txt"
    ex.write_text(PAG + "\n")
    a = _parser().parse_args(["distill", str(src), "--model", "m", "--exclude-ids", str(ex),
                              "--out-resolved", str(r), "--out-failed", str(f)])
    assert a.fn(a) == 2

    out = tmp_path / "mix.jsonl"
    a = _parser().parse_args(["mix", "--teacher", str(r), "--engineer", str(src), "--ratio", "1:1",
                              "--out", str(out)])
    assert a.fn(a) == 0
    mixed = load_demos(out)
    assert sorted(d.metadata["mix_source"] for d in mixed) == ["engineer", "teacher"]
