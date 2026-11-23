"""§3a SFT mix: rendering + context budget, loss masks, filters, backfill, no-op collapse, consistency."""
import argparse
import copy
import json
from pathlib import Path

import pytest

from repo_fix_agent import sandbox as sb
from repo_fix_agent import sft
from repo_fix_agent.pipeline import sft_data as S
from repo_fix_agent.policies import StudentServingConfig, apply_context_budget, LocalModelPolicy
from repo_fix_agent.schema import Action, Demo, Step, save_demos

PAGINATION = '''def paginate(items, page, per_page):
    """Return the items on a 1-indexed page."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be >= 1")
    start = page * per_page
    return items[start:start + per_page]


def total_pages(n_items, per_page):
    return (n_items + per_page - 1) // per_page
'''
EXISTING_TESTS = '''from webapp.pagination import total_pages


def test_total_pages():
    assert total_pages(25, 10) == 3
'''
PATCH = '''diff --git a/webapp/pagination.py b/webapp/pagination.py
--- a/webapp/pagination.py
+++ b/webapp/pagination.py
@@ -2,7 +2,7 @@ def paginate(items, page, per_page):
     """Return the items on a 1-indexed page."""
     if page < 1 or per_page < 1:
         raise ValueError("page and per_page must be >= 1")
-    start = page * per_page
+    start = (page - 1) * per_page
     return items[start:start + per_page]


'''
TEST_PATCH = '''diff --git a/tests/test_issue_001.py b/tests/test_issue_001.py
new file mode 100644
--- /dev/null
+++ b/tests/test_issue_001.py
@@ -0,0 +1,5 @@
+from webapp.pagination import paginate
+
+
+def test_first_page_starts_at_zero():
+    assert paginate(list(range(25)), 1, 10) == list(range(10))
'''
CMD = "python -m pytest -q -p no:cacheprovider"


def _step(thought, tool, args, obs=""):
    return Step(thought, Action(tool, args), obs)


def make_demo(root: Path, iid="inst-001") -> Demo:
    repo = root / "repo"
    (repo / "webapp").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "webapp" / "__init__.py").write_text("")
    (repo / "webapp" / "pagination.py").write_text(PAGINATION)
    (repo / "tests" / "test_pagination.py").write_text(EXISTING_TESTS)
    traj = [
        _step("Find where paginate is defined.", "bash", {"command": "grep -rn 'def paginate' ."},
              "./webapp/pagination.py:1:def paginate(items, page, per_page):\n\n[exit code 0]"),
        _step("Read the implementation.", "view_file", {"path": "webapp/pagination.py"}, PAGINATION),
        _step("The offset treats page as 0-indexed; for 1-indexed pages use (page - 1).", "str_replace",
              {"path": "webapp/pagination.py", "old_str": "start = page * per_page",
               "new_str": "start = (page - 1) * per_page"}, "Edited webapp/pagination.py."),
        _step("Run the existing tests for regressions.", "bash", {"command": CMD}, "1 passed\n\n[exit code 0]"),
        _step("Fix verified.", "submit", {}, "Submitted."),
    ]
    return Demo(iid, "repo", None, "Pagination skips the first page.\npaginate(items, 1, 10) returns items 10-19.",
                CMD, TEST_PATCH, ["tests/test_issue_001.py::test_first_page_starts_at_zero"],
                ["tests/test_pagination.py::test_total_pages"], PATCH, traj,
                {"source": "engineer", "replay_ok": True})


@pytest.fixture(autouse=True)
def _local_sandbox(monkeypatch):
    monkeypatch.setattr(sb, "_docker_available", lambda: False)
    # once-per-process warning flags: restore them so test_sandbox's log assertions stay order-independent
    monkeypatch.setattr(sb, "_warned_fallback", sb._warned_fallback)
    monkeypatch.setattr(sb, "_warned_network", sb._warned_network)


@pytest.fixture
def demo(tmp_path):
    return make_demo(tmp_path)


def teacher_copy(d: Demo, k: int, resolved=True) -> Demo:
    t = copy.deepcopy(d)
    t.instance_id = f"{d.instance_id}#t{k}"
    t.metadata = {"source": "teacher:claude-opus-5-5", "resolved": resolved, "parent_id": d.instance_id}
    return t


# ---- rendering / masks / budget ---------------------------------------------------------------
def test_record_has_loss_mask_on_assistant_only(demo):
    rec = S.render_record(demo)
    assert set(rec) == {"id", "messages", "tools", "loss_mask", "weight", "source"}
    assert len(rec["loss_mask"]) == len(rec["messages"])
    assert [m for m, msg in zip(rec["loss_mask"], rec["messages"]) if m] and all(
        (msg["role"] == "assistant") == m for m, msg in zip(rec["loss_mask"], rec["messages"]))
    assert rec["tools"] == sft.tools_as_functions()
    cfg0 = StudentServingConfig()
    assert rec["messages"] == sft.render_messages(demo, cfg0.agent_options().system_prompt(),
                                                  reasoning_field=cfg0.keep_reasoning)  # nothing elided


def test_context_budget_is_the_shared_policy(demo):
    cfg = StudentServingConfig(max_new_tokens=100, chars_per_token=1.0, keep_recent_observations=1)
    base = S.message_tokens([]) + len(json.dumps(sft.tools_as_functions()))
    cfg.max_context_tokens = base + cfg.max_new_tokens + 1200
    rec = S.render_record(demo, cfg)
    assert rec["messages"] == apply_context_budget(
        sft.render_messages(demo, cfg.agent_options().system_prompt(), reasoning_field=cfg.keep_reasoning),
        sft.tools_as_functions(), cfg)
    assert any(m["content"] == LocalModelPolicy.ELIDED for m in rec["messages"] if m["role"] == "tool")


# ---- filters / mix ----------------------------------------------------------------------------
def test_mix_filters_and_ratio(demo):
    bad_eng = copy.deepcopy(demo)
    bad_eng.instance_id, bad_eng.metadata = "bad", {"replay_ok": False}
    teachers = [teacher_copy(demo, k) for k in range(6)] + [teacher_copy(demo, 9, resolved=False)]
    recs = S.build_sft_mix([demo, bad_eng], teachers, ratio=(1, 3))
    srcs = [r["source"] for r in recs]
    assert srcs.count("engineer") == 1 and sum(s.startswith("teacher:") for s in srcs) == 3
    assert "bad" not in {r["id"] for r in recs}
    assert len({r["id"] for r in recs}) == len(recs)
    # engineer only: all kept
    assert len(S.build_sft_mix([demo, bad_eng])) == 1


def test_aux_capped_by_tokens(demo):
    aux = [{"id": f"a{i}", "task": "localization", "meta": {"instance_id": "x"},
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u" * 60},
                         {"role": "assistant", "content": "a" * 40}]} for i in range(50)]
    recs = S.build_sft_mix([demo], aux=aux, aux_frac=0.2)
    aux_tok = sum(S.message_tokens(r["messages"]) for r in recs if r["source"].startswith("aux:"))
    total = sum(S.message_tokens(r["messages"]) for r in recs)
    assert 0 < aux_tok <= 0.2 * total + 1
    a = next(r for r in recs if r["source"] == "aux:localization")
    assert a["loss_mask"] == [False, False, True]


def test_exclude_ids_refused(demo, tmp_path):
    with pytest.raises(ValueError, match="excluded"):
        S.build_sft_mix([demo], exclude_ids={demo.instance_id})
    with pytest.raises(ValueError, match="excluded"):  # via parent_id
        S.build_sft_mix([], [teacher_copy(demo, 0)], exclude_ids={demo.instance_id})
    m = tmp_path / "frozen.json"
    m.write_text(json.dumps({"instances": {"a": "h1", "b": "h2"}}))
    assert S.load_exclude_ids(m) == {"a", "b"}
    m.write_text(json.dumps({"instances": ["c", {"instance_id": "d"}]}))
    assert S.load_exclude_ids(m) == {"c", "d"}
    m.write_text("e\n\n# comment\nf\n")
    assert S.load_exclude_ids(m) == {"e", "f"}


# ---- backfill ---------------------------------------------------------------------------------
def test_backfill_rejects_early_gold_leak_and_hidden_tests(demo):
    blank = copy.deepcopy(demo)
    for st in blank.trajectory:
        st.thought = ""
    leaky = lambda system, prompt: "The fix is start = (page - 1) * per_page."  # noqa: E731
    out = S.backfill_thoughts(blank, leaky)
    # steps 0 and 1 happen before webapp/pagination.py was viewed -> rejected, thought kept
    assert out.trajectory[0].thought == "" and out.trajectory[1].thought == ""
    assert out.trajectory[2].thought.startswith("The fix")  # after view_file: allowed
    assert out.metadata["backfill"] == {"accepted": 3, "rejected": 2}
    names = lambda s, p: "Make test_first_page_starts_at_zero pass."  # noqa: E731
    assert S.backfill_thoughts(blank, names).metadata["backfill"] == {"accepted": 0, "rejected": 5}
    empty = lambda s, p: "  "  # noqa: E731
    assert S.backfill_thoughts(blank, empty).metadata["backfill"]["rejected"] == 5


def test_backfill_clean_accepted_and_prompt_has_fixed_action(demo):
    blank = copy.deepcopy(demo)
    blank.trajectory[1].thought = "read"
    seen = []

    def clean(system, prompt):
        seen.append(prompt)
        return "Open the module to see how the start offset is computed."

    out = S.backfill_thoughts(blank, clean, min_chars=10)
    assert out.metadata["backfill"] == {"accepted": 1, "rejected": 0}
    assert out.trajectory[1].thought.startswith("Open the module")
    assert out.trajectory[0].thought == demo.trajectory[0].thought  # long enough: untouched
    assert '"view_file"' in seen[0] and "grep -rn" in seen[0]  # fixed next action + prior step
    assert demo.trajectory[1].thought == "Read the implementation."  # input not mutated


# ---- collapse no-ops --------------------------------------------------------------------------
def test_collapse_noops_keeps_patch(demo, tmp_path):
    noisy = copy.deepcopy(demo)
    t = noisy.trajectory
    noisy.trajectory = [t[0], copy.deepcopy(t[0]),
                        _step("typo", "bash", {"command": "cat webapp/paginaton.py"},
                              "cat: webapp/paginaton.py: No such file or directory\n\n[exit code 1]"),
                        _step("retry", "bash", {"command": "cat webapp/pagination.py"},
                              PAGINATION + "\n[exit code 0]")] + t[1:]
    assert S.find_noops(noisy.trajectory) == [1, 2]
    out = S.collapse_noops(noisy, tmp_path)
    assert len(out.trajectory) == len(noisy.trajectory) - 2
    assert out.metadata["collapsed_steps"] == 2
    assert [s.action for s in out.trajectory] == [s.action for i, s in enumerate(noisy.trajectory) if i not in (1, 2)]
    assert "[exit code 0]" in out.trajectory[0].observation  # re-executed observations


def test_collapse_noops_reverts_when_patch_changes(demo, tmp_path):
    noisy = copy.deepcopy(demo)
    # a "failed" command that actually has a side effect the patch depends on
    side = _step("x", "bash", {"command": "echo '# note' >> webapp/pagination.py; exit 3"}, "[exit code 3]")
    retry = _step("y", "bash", {"command": "echo '# note' >> webapp/pagination.py; exit 0"}, "[exit code 0]")
    noisy.trajectory = [side, retry] + noisy.trajectory
    with S.RepoEnv(str(tmp_path / "repo"), backend="local") as env:
        for st in noisy.trajectory:
            env.execute(st.action)
        noisy.patch = env.diff()
    assert S.find_noops(noisy.trajectory) == [0]
    assert S.collapse_noops(noisy, tmp_path) is noisy


# ---- consistency ------------------------------------------------------------------------------
def test_prompt_consistency_with_local_model_policy(demo):
    assert S.check_prompt_consistency(demo, StudentServingConfig(), steps=3) == []
    cfg = StudentServingConfig(max_new_tokens=100, chars_per_token=1.0, keep_recent_observations=1)
    cfg.max_context_tokens = len(json.dumps(sft.tools_as_functions())) + 1300
    assert S.check_prompt_consistency(demo, cfg, steps=4) == []


def test_prompt_consistency_detects_drift(demo, monkeypatch):
    real = sft.render_messages
    monkeypatch.setattr(sft, "render_messages", lambda d, *a, **kw: [{**m, "content": m["content"] + "!"}
                                                                     if m["role"] == "system" else m
                                                                     for m in real(d, *a, **kw)])
    assert "system prompt differs" in S.check_prompt_consistency(demo)


# ---- CLI --------------------------------------------------------------------------------------
def test_cli_sft_mix(demo, tmp_path, capsys):
    eng, tea, out = tmp_path / "e.jsonl", tmp_path / "t.jsonl", tmp_path / "out" / "sft.jsonl"
    save_demos([demo], eng)
    save_demos([teacher_copy(demo, k) for k in range(3)], tea)
    p = argparse.ArgumentParser()
    S.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["sft-mix", "--engineer", str(eng), "--teacher", str(tea), "--out", str(out),
                      "--ratio", "1:3", "--config", str(tmp_path / "missing.yaml")])
    assert a.fn(a) == 0
    recs = S.read_jsonl(out)
    assert len(recs) == 4 and json.loads(capsys.readouterr().out)["n"] == 4
    ex = tmp_path / "ex.txt"
    ex.write_text(demo.instance_id + "\n")
    a = p.parse_args(["sft-mix", "--engineer", str(eng), "--out", str(out), "--exclude-ids", str(ex)])
    assert a.fn(a) == 2  # frozen-test violation -> clean error, exit 2
