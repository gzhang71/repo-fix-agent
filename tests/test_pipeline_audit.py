import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from repo_fix_agent import grader, sandbox  # noqa: E402
from repo_fix_agent.grader import GradeResult  # noqa: E402
from repo_fix_agent.pipeline import audit  # noqa: E402
from repo_fix_agent.schema import Action, Demo, Step, load_demos  # noqa: E402

SEED = ROOT / "data" / "demos" / "seed.jsonl"


def _load_seed(n=4):
    """Robust read: seed.jsonl may be regenerated concurrently; take the first few original seeds."""
    out = []
    for line in SEED.read_text().splitlines():
        try:
            d = Demo.from_dict(json.loads(line))
        except Exception:
            continue
        if d.instance_id.startswith("seed-0") and (ROOT / d.repo).is_dir():
            out.append(d)
    out.sort(key=lambda d: d.instance_id)
    if len(out) < n:
        pytest.skip("seed demos unavailable")
    return out[:n]


@pytest.fixture(autouse=True)
def _keep_sandbox_warning_state(monkeypatch):
    # real grading with backend="auto" flips sandbox's one-shot fallback-warning flag; restore it
    # so tests/test_sandbox.py::test_auto_falls_back_to_local still sees the warning.
    monkeypatch.setattr(sandbox, "_warned_fallback", getattr(sandbox, "_warned_fallback", False),
                        raising=False)


@pytest.fixture(scope="module")
def seeds():
    return _load_seed()


@pytest.fixture
def demos(seeds):
    return copy.deepcopy(seeds)


def fake_grade(demo, patch, root="."):
    fixed = bool(patch.strip()) and audit.patches_equivalent(patch, demo.patch)
    return GradeResult(fixed, {t: fixed for t in demo.FAIL_TO_PASS}, {t: True for t in demo.PASS_TO_PASS})


# ---------------------------------------------------------------- validation / flakiness
def test_validate_n_real_seed(demos):
    ok, flaky, problems = audit.validate_n(demos[0], ROOT, runs=1)
    assert ok and not flaky and problems == []


def test_validate_n_invalid_with_real_grader(demos):
    d = demos[0]
    d.patch = ""
    ok, flaky, problems = audit.validate_n(d, ROOT, runs=1)
    assert not ok and not flaky
    assert any("does not resolve" in p for p in problems)


def test_validate_n_detects_flaky(demos, monkeypatch):
    calls = {"n": 0}

    def flaky_grade(demo, patch, root="."):
        r = fake_grade(demo, patch, root)
        if patch:
            calls["n"] += 1
            if calls["n"] == 2:  # second run: one F2P test flips
                first = demo.FAIL_TO_PASS[0]
                r.fail_to_pass[first] = False
                r.resolved = False
        return r

    monkeypatch.setattr(grader, "grade", flaky_grade)
    ok, flaky, problems = audit.validate_n(demos[0], ROOT, runs=3)
    assert flaky and not ok
    assert any(p.startswith("run 2") for p in problems)


def test_validate_n_grading_exception_is_invalid(demos):
    def boom(*a, **k):
        raise FileNotFoundError("repo not found")
    ok, flaky, problems = audit.validate_n(demos[0], ROOT, runs=2, grade_fn=boom)
    assert not ok and not flaky and "FileNotFoundError" in problems[0]


# ---------------------------------------------------------------- replay
def test_replay_real_and_broken(demos):
    d = demos[0]
    assert audit.replay_check(d, ROOT)
    broken = copy.deepcopy(d)
    broken.trajectory = [s for s in broken.trajectory if s.action.tool != "str_replace"]
    r = audit.replay(broken, ROOT)
    assert not r.ok and r.diff.strip() == ""


def test_patches_equivalent_ignores_whitespace_and_headers(demos):
    p = demos[0].patch
    variant = p.replace("index 466ab88..1b92fcd 100644\n", "").replace("    start", "\tstart")
    variant = "\n".join(l if not l.startswith("@@") else "@@ -1,7 +1,7 @@" for l in variant.splitlines())
    assert audit.patches_equivalent(p, variant)
    assert not audit.patches_equivalent(p, p.replace("(page - 1)", "(page - 2)"))


# ---------------------------------------------------------------- dedup
def test_exact_dedup_patch_and_issue(demos):
    a, b = demos[0], demos[1]
    dup_patch = copy.deepcopy(a)
    dup_patch.instance_id = "zz-dup-patch"
    dup_patch.problem_statement = "something else entirely"
    dup_patch.patch = a.patch.replace("    start", "  start")  # whitespace-only change
    dup_issue = copy.deepcopy(b)
    dup_issue.instance_id = "zz-dup-issue"
    dup_issue.problem_statement = "  " + b.problem_statement.upper() + "\n\n"
    dup_issue.patch = b.patch.replace("tags", "labels")
    kept, dropped = audit.exact_dedup([a, b, dup_patch, dup_issue])
    assert [d.instance_id for d in kept] == [a.instance_id, b.instance_id]
    assert ("zz-dup-patch", a.instance_id, "patch") in dropped
    assert ("zz-dup-issue", b.instance_id, "problem_statement") in dropped


def test_near_dup_clusters(demos):
    a = demos[0]
    near = copy.deepcopy(a)
    near.instance_id = "seed-001-zz-cherry-pick"
    near.problem_statement = a.problem_statement.replace("Pages are", "Note: pages are")
    clusters = audit.near_dup_clusters(demos + [near], threshold=0.7)
    assert clusters[a.instance_id] == clusters[near.instance_id] == f"cluster:{a.instance_id}"
    others = {clusters[d.instance_id] for d in demos[1:]}
    assert len(others) == len(demos) - 1 and clusters[a.instance_id] not in others


def test_near_dup_lsh_path_matches_pairwise(demos):
    a = demos[0]
    near = copy.deepcopy(a)
    near.instance_id = "zz-near"
    near.problem_statement += " Thanks!"
    ds = demos + [near]
    pair = audit.near_dup_clusters(ds, threshold=0.7)
    lsh = audit.near_dup_clusters(ds, threshold=0.7, pairwise_max=0)
    assert pair == lsh


def test_minhash_estimate_tracks_jaccard():
    s1 = audit.shingles(" ".join(f"w{i}" for i in range(200)))
    s2 = audit.shingles(" ".join(f"w{i}" for i in range(100, 300)))
    true = len(s1 & s2) / len(s1 | s2)
    est = audit.estimate_jaccard(audit.minhash_signature(s1, 256), audit.minhash_signature(s2, 256))
    assert abs(est - true) < 0.1


# ---------------------------------------------------------------- sizes
MULTI = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
@@ -10,2 +10,3 @@
 y = 1
+z = 3
"""


def test_patch_size_buckets(demos):
    assert audit.patch_size(demos[0].patch) == {"lines": 2, "files": 1, "hunks": 1, "bucket": "single-hunk"}
    assert audit.patch_size(MULTI) == {"lines": 3, "files": 1, "hunks": 2, "bucket": "multi-hunk"}
    two = MULTI + MULTI.replace("a.py", "b.py")
    assert audit.patch_size(two)["bucket"] == "multi-file"
    assert audit.patch_size(two)["files"] == 2


def test_trajectory_length_flags(demos):
    d = demos[0]
    assert audit.trajectory_stats(d)["flags"] == []
    d.trajectory = d.trajectory * 7
    d.trajectory[0] = Step("t", Action("bash", {"command": "cat big"}), "x" * 300_000)
    st = audit.trajectory_stats(d)
    assert "steps>40" in st["flags"] and "tokens>64000" in st["flags"]
    assert st["approx_tokens"] > 300_000 / 3.5 - 1


# ---------------------------------------------------------------- leakage
def test_seed_has_no_leakage_false_positives(demos):
    # seed issues contain legitimate repro snippets that share single lines with tests
    for d in demos:
        assert audit.detect_leakage(d) == [], d.instance_id


def test_leakage_in_issue(demos):
    d = demos[0]
    d.problem_statement += ("\nSee test_first_page_starts_at_zero in tests/test_issue_001.py:\n"
                            "    assert paginate(list(range(25)), 1, 10) == list(range(10))\n")
    f = audit.detect_leakage(d)
    assert any("hidden assertion" in x for x in f)
    assert any("FAIL_TO_PASS test: test_first_page_starts_at_zero" in x for x in f)
    assert any("hidden test file: tests/test_issue_001.py" in x for x in f)


def test_leakage_trajectory_reads_and_removal(demos):
    d = demos[0]
    leak_view = Step("peek at the hidden test", Action("view_file", {"path": "tests/test_issue_001.py"}), "...")
    leak_cat = Step("cat it", Action("bash", {"command": "cat tests/test_issue_001.py"}), "...")
    d.trajectory = [leak_view] + d.trajectory[:3] + [leak_cat] + d.trajectory[3:]
    assert audit.hidden_test_read_steps(d) == [0, 4]
    f = audit.detect_leakage(d)
    assert sum("reads hidden test file" in x for x in f) == 2
    clean = audit.remove_hidden_test_reads(d)
    assert len(clean.trajectory) == len(d.trajectory) - 2
    assert "replay_ok" not in clean.metadata and clean.metadata["removed_hidden_test_steps"] == [0, 4]
    assert audit.hidden_test_read_steps(clean) == []
    assert len(d.trajectory) == len(clean.trajectory) + 2  # input not mutated
    replayed = audit.remove_hidden_test_reads(d, root=ROOT)
    assert replayed.metadata["replay_ok"] is True


def test_own_created_test_file_is_not_leak(demos):
    d = demos[0]
    d.trajectory = [Step("write repro", Action("create_file", {"path": "tests/test_issue_001.py", "content": "x"}), "ok"),
                    Step("run it", Action("bash", {"command": "python -m pytest tests/test_issue_001.py"}), "ok")]
    assert audit.hidden_test_read_steps(d) == []
    assert any("creates a file at a hidden test path" in x for x in audit.detect_leakage(d))


# ---------------------------------------------------------------- secrets
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"


def test_secret_scan_and_redact(demos):
    d = demos[0]
    assert audit.scan_secrets(d) == []
    d.problem_statement += f"\nContact jane.doe@corp.io from 10.1.2.3 with key {AWS}\n"
    d.trajectory[0].observation += f'\napi_key = "sk_live_abcdef123456"\nAuthorization: Bearer abcdefghijklmnop1234\n{PEM}'
    findings = audit.scan_secrets(d)
    for kind in ("email in problem_statement", "ip_address in problem_statement",
                 "aws_access_key in problem_statement", "api_key in trajectory[0].observation",
                 "bearer_token in trajectory[0].observation", "private_key in trajectory[0].observation"):
        assert kind in findings, kind
    assert not any(AWS in f for f in findings)  # never echo the secret
    r = audit.redact(d)
    assert audit.scan_secrets(r) == []
    assert AWS not in r.problem_statement and "[REDACTED:aws_access_key]" in r.problem_statement
    assert "MIIEow" not in r.trajectory[0].observation
    assert AWS in d.problem_statement  # original untouched


def test_secret_scan_avoids_common_false_positives():
    text = ('token = get_token()\npassword = os.environ["PW"]\nhost 127.0.0.1\nsee user@example.com\n'
            "version 1.2.3")
    assert audit.scan_text(text) == []


# ---------------------------------------------------------------- LLM labels
def test_bug_type_and_sufficiency_with_fake_llm(demos):
    seen = []

    def llm(system, prompt):
        seen.append(prompt)
        if "Labels:" in prompt:
            return "Reasoning... the start index is shifted by one page.\nLABEL: Off by one"
        return "The issue states the expected output.\nVERDICT: **SUFFICIENT**"

    assert audit.label_bug_type(demos[0], llm) == "off-by-one"
    assert audit.judge_issue_sufficiency(demos[0], llm) is True
    assert "paginate" in seen[0]


@pytest.mark.parametrize("answer,expected", [
    ('{"bug_type": "null_handling"}', "null-handling"),
    ("bug type: concurrency", "concurrency"),
    ("I think this is a wrong-operator bug", "wrong-operator"),
    ("no idea", "other"),
])
def test_parse_bug_type(answer, expected):
    assert audit.parse_bug_type(answer) == expected


@pytest.mark.parametrize("answer,expected", [
    ("VERDICT: INSUFFICIENT", False), ("verdict: sufficient", True), ('{"sufficient": false}', False),
    ("The issue is underspecified.", False), ("Yes.", True), ("hmm", None),
])
def test_parse_sufficiency(answer, expected):
    assert audit.parse_sufficiency(answer) is expected


def test_judge_survives_llm_errors(demos):
    def bad(system, prompt):
        raise RuntimeError("refused")
    assert audit.judge_issue_sufficiency(demos[0], bad) is None


# ---------------------------------------------------------------- driver + report
def _variant(d, iid, statement=None, distinct_patch=True):
    v = copy.deepcopy(d)
    v.instance_id = iid
    if statement is not None:
        v.problem_statement = statement
    if distinct_patch:  # different normalized patch hash -> not an exact duplicate
        v.patch = v.patch.rstrip("\n") + f"\n+# distinct {iid}\n"
    return v


def _variants(demos):
    b = demos
    invalid = _variant(b[1], "x-invalid", "invalid variant", distinct_patch=False)
    invalid.patch = ""
    flaky = _variant(b[2], "x-flaky", "flaky variant")
    dup = _variant(b[0], "x-dup", "dup variant text", distinct_patch=False)
    near = _variant(b[3], "x-near", b[3].problem_statement + " (also seen on the release branch)")
    leaky = _variant(b[2], "x-leaky", b[2].problem_statement + "\nRelevant: " + " ".join(
        n.split("::")[-1] for n in b[2].FAIL_TO_PASS))
    secret = _variant(b[1], "x-secret", b[1].problem_statement + f"\nmy key is {AWS}")
    return [invalid, flaky, dup, near, leaky, secret]


def test_audit_demos_end_to_end(demos, monkeypatch):
    vs = _variants(demos)
    flips = {"n": 0}

    def grade(demo, patch, root="."):
        r = fake_grade(demo, patch, root)
        if demo.instance_id == "x-flaky" and patch:
            flips["n"] += 1
            r.resolved = flips["n"] % 2 == 0
            r.fail_to_pass = {t: r.resolved for t in demo.FAIL_TO_PASS}
        return r

    replayed = []

    def replay_fn(d, root):
        replayed.append(d.instance_id)
        return d.instance_id != "x-secret"

    def llm(system, prompt):
        return "LABEL: logic-error" if "Labels:" in prompt else "VERDICT: INSUFFICIENT"

    inp = demos + vs
    snapshot = copy.deepcopy(inp)
    kept, rep = audit.audit_demos(inp, root=ROOT, runs=3, llm=llm, near_dup_threshold=0.7,
                                  grade_fn=grade, replay_fn=replay_fn,
                                  public_map={demos[0].repo: True})
    ids = [d.instance_id for d in kept]
    assert "x-invalid" not in ids and "x-flaky" not in ids and "x-dup" not in ids
    assert {"x-near", "x-leaky", "x-secret"} <= set(ids)
    assert [n for _, n in rep.stages] == [len(inp) - 1, len(inp) - 2, len(inp) - 3]
    reasons = {d["instance_id"]: d["reason"] for d in rep.dropped}
    assert reasons == {"x-invalid": "invalid", "x-flaky": "flaky", "x-dup": "exact-duplicate"}
    by = {d.instance_id: d for d in kept}
    assert by["x-leaky"].metadata["leakage"] and rep.flags["leakage"] == ["x-leaky"]
    assert by["x-secret"].metadata["secrets"] and rep.flags["secrets"] == ["x-secret"]
    assert rep.flags["replay_failed"] == ["x-secret"] and by["x-secret"].metadata["replay_ok"] is False
    assert set(replayed) == set(ids)  # replay only kept demos
    assert by["x-near"].metadata["cluster_id"] == by[demos[3].instance_id].metadata["cluster_id"]
    assert rep.clusters[f"cluster:{demos[3].instance_id}"] == sorted([demos[3].instance_id, "x-near"])
    assert by[demos[0].instance_id].metadata["cluster_id"] == f"cluster:{demos[0].instance_id}"
    d0 = by[demos[0].instance_id].metadata
    assert d0["public"] is True and by[demos[1].instance_id].metadata["public"] is None
    assert d0["flaky"] is False and d0["replay_ok"] is True
    assert d0["patch_size"]["bucket"] == "single-hunk" and d0["repo_group"] == demos[0].repo
    assert d0["bug_type"] == "off-by-one"  # existing labels kept
    assert d0["issue_sufficient"] is False
    assert sum(rep.distributions["patch_size"].values()) == len(kept)
    # per-repo: base[1]'s repo had 3 instances (base, invalid, secret) -> 1/3 dropped > 20%
    r1 = rep.per_repo[demos[1].repo]
    assert r1["total"] == 3 and r1["invalid"] == 1 and abs(r1["drop_rate"] - 1 / 3) < 1e-9
    assert any(demos[1].repo in w and "fix the env" in w for w in rep.warnings)
    assert inp[0].metadata == snapshot[0].metadata  # input not mutated

    md = rep.to_markdown()
    for s in ("Counts after each filter", "Per-repo table", "Warnings", "patch_size", "bug_type",
              "x-flaky", "flaky", "exact-duplicate", "Near-duplicate clusters", "x-near", "replay_failed"):
        assert s in md, s
    assert AWS not in md


def test_audit_runs_zero_skips_validation(demos):
    kept, rep = audit.audit_demos(demos, runs=0, replay_trajectories=False)
    assert len(kept) == len(demos)
    assert all(d.metadata["flaky"] is None and "replay_ok" not in d.metadata for d in kept)


def test_cli_audit(demos, tmp_path, monkeypatch):
    src = tmp_path / "in.jsonl"
    from repo_fix_agent.schema import save_demos
    dup = copy.deepcopy(demos[0]); dup.instance_id = "zz-dup"
    save_demos(demos[:2] + [dup], src)
    monkeypatch.setattr(grader, "grade", fake_grade)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    audit.register(sub)
    out, rep = tmp_path / "o" / "v.jsonl", tmp_path / "o" / "audit.md"
    a = p.parse_args(["audit", str(src), "--out", str(out), "--report", str(rep), "--root", str(ROOT),
                      "--runs", "2", "--no-replay"])
    assert a.fn(a) == 0
    kept = load_demos(out)
    assert [d.instance_id for d in kept] == [d.instance_id for d in demos[:2]]
    assert "exact-duplicate" in rep.read_text()
