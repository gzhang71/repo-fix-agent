import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from repo_fix_agent.pipeline import split as sp  # noqa: E402
from repo_fix_agent.schema import Demo, load_demos, save_demos  # noqa: E402

SEED = ROOT / "data" / "demos" / "seed.jsonl"
BUG_TYPES = ["off-by-one", "logic-error", "error-handling", "state-update"]
BUCKETS = ["single-hunk", "multi-hunk", "multi-file"]


@pytest.fixture(scope="module")
def seed():
    for line in SEED.read_text().splitlines():
        try:
            d = Demo.from_dict(json.loads(line))
        except Exception:
            continue
        if d.instance_id.startswith("seed-0"):
            return d
    pytest.skip("seed demos unavailable")


def make(seed, n, repos=None, **md):
    """n synthetic instances derived from one seed record, each its own repo by default."""
    out = []
    for i in range(n):
        d = copy.deepcopy(seed)
        d.instance_id = f"inst-{i:03d}"
        d.repo = repos[i] if repos else f"data/repos/r{i:03d}"
        d.problem_statement = f"Issue number {i}: {seed.problem_statement}"
        d.patch = seed.patch.replace("(page - 1)", f"(page - {i + 1})")
        d.metadata = {"bug_type": BUG_TYPES[i % 4], "difficulty": ["easy", "medium"][i % 2],
                      "patch_size": {"lines": 2, "files": 1, "hunks": 1, "bucket": BUCKETS[i % 3]},
                      "language": "python", "flaky": False, "leakage": [], **md}
        out.append(d)
    return out


def ids(ds):
    return [d.instance_id for d in ds]


# ---------------------------------------------------------------- split
def test_small_n_every_instance_own_repo(seed):
    ds = make(seed, 10)
    s = sp.split_demos(ds, seed=0)
    assert [len(s[k]) for k in sp.SPLITS] == [8, 1, 1]
    assert sorted(ids(s["train"] + s["dev"] + s["test"])) == ids(ds)
    assert sp.check_split_integrity(s) == []


def test_split_deterministic_and_ratios(seed):
    ds = make(seed, 60)
    a = sp.split_demos(ds, (0.8, 0.1, 0.1), seed=3)
    b = sp.split_demos(list(reversed(copy.deepcopy(ds))), (0.8, 0.1, 0.1), seed=3)
    assert {k: ids(v) for k, v in a.items()} == {k: ids(v) for k, v in b.items()}
    assert len(a["test"]) == 6 and len(a["dev"]) == 6 and len(a["train"]) == 48
    others = [sp.split_demos(ds, seed=k) for k in range(4)]
    assert len({tuple(ids(o["test"])) for o in others}) > 1  # seed matters


def test_split_stratifies_eval_splits(seed):
    ds = make(seed, 80)
    s = sp.split_demos(ds, seed=0)
    for k in ("dev", "test"):
        st = [sp.strata_of(d) for d in s[k]]
        assert {x["bug_type"] for x in st} == set(BUG_TYPES)
        assert {x["patch_size"] for x in st} == set(BUCKETS)
        assert {x["difficulty"] for x in st} == {"easy", "medium"}


def test_repo_groups_and_forks_stay_together(seed):
    repos = [f"data/repos/r{i // 3:02d}" for i in range(30)]  # 3 instances per repo
    ds = make(seed, 30, repos=repos)
    for d in ds:  # r00 and r01 are a fork pair
        if d.repo in ("data/repos/r00", "data/repos/r01"):
            d.metadata["repo_group"] = "upstream/r00"
    s = sp.split_demos(ds, seed=1)
    assert sp.check_split_integrity(s) == []
    where = {d.instance_id: k for k, v in s.items() for d in v}
    fork = [d.instance_id for d in ds if d.metadata.get("repo_group") == "upstream/r00"]
    assert len({where[i] for i in fork}) == 1
    assert s["dev"] and s["test"]


def test_near_dup_clusters_never_straddle(seed):
    ds = make(seed, 20)
    for i in (0, 7, 13, 19):  # same fix cherry-picked to four repos
        ds[i].metadata["cluster_id"] = "cluster:inst-000"
    for k in range(6):
        s = sp.split_demos(ds, seed=k)
        assert sp.check_split_integrity(s) == []
        where = {d.instance_id: name for name, v in s.items() for d in v}
        assert len({where[f"inst-{i:03d}"] for i in (0, 7, 13, 19)}) == 1


def test_integrity_check_reports_violations(seed):
    ds = make(seed, 4)
    ds[0].metadata["cluster_id"] = ds[1].metadata["cluster_id"] = "c"
    bad = {"train": [ds[0], ds[2]], "dev": [ds[1]], "test": [ds[3]]}
    assert any("cluster 'c'" in p for p in sp.check_split_integrity(bad))


def test_frozen_ids_forced_into_test(seed):
    ds = make(seed, 30)
    s = sp.split_demos(ds, seed=0, frozen_ids={"inst-005", "inst-017"})
    assert {"inst-005", "inst-017"} <= set(ids(s["test"]))


def test_tiny_and_degenerate_inputs(seed):
    assert sp.split_demos([]) == {"train": [], "dev": [], "test": []}
    one = sp.split_demos(make(seed, 1))
    assert len(one["train"]) == 1
    two = sp.split_demos(make(seed, 2))
    assert len(two["train"]) >= 1 and len(two["train"]) + len(two["dev"]) + len(two["test"]) == 2
    with pytest.raises(ValueError):
        sp.split_demos(make(seed, 3), ratios=(0.5, 0.5))


def test_time_split(seed):
    repos = ["data/repos/a"] * 5 + ["data/repos/b"] * 2 + ["data/repos/c"]
    ds = make(seed, 8, repos=repos)
    for i, d in enumerate(ds):
        d.metadata["created_at"] = f"2024-01-{i + 1:02d}"
    t = sp.time_split(ds, test_frac=0.2)
    assert ids(t["in_repo_test"]) == ["inst-004", "inst-006"]  # newest of a, newest of b; c alone
    assert len(t["in_repo_train"]) == 6


# ---------------------------------------------------------------- frozen manifest
def test_manifest_roundtrip_verify_and_tamper(seed, tmp_path):
    ds = make(seed, 10)
    test = ds[:3]
    path = tmp_path / "frozen.json"
    man = sp.write_frozen_manifest(test, path, created="2026-01-01")
    assert man["count"] == 3 and sp.load_frozen_ids(path) == set(ids(test))
    assert sp.verify_frozen(path, ds).ok
    tampered = copy.deepcopy(ds)
    tampered[1].test_patch += "\n# weakened"
    del tampered[2]
    chk = sp.verify_frozen(path, tampered)
    assert not chk.ok and chk.modified == ["inst-001"] and chk.missing == ["inst-002"]
    assert "modified" in str(chk)
    # same set again: keeps original created date; different set: refuses unless overwrite
    again = sp.write_frozen_manifest(test, path, created="2030-01-01")
    assert again["created"] == "2026-01-01"
    with pytest.raises(FileExistsError):
        sp.write_frozen_manifest(ds[3:5], path)
    sp.write_frozen_manifest(ds[3:5], path, overwrite=True)
    assert sp.load_frozen_ids(path) == {"inst-003", "inst-004"}


def test_assert_disjoint(seed, tmp_path):
    ds = make(seed, 6)
    path = tmp_path / "frozen.json"
    sp.write_frozen_manifest(ds[:2], path)
    sp.assert_disjoint(ds[2:], path, "SFT")  # fine
    sp.assert_disjoint(ds[2:], {"inst-000"}, "SFT")
    with pytest.raises(sp.FrozenLeakError, match=r"1 frozen test instance\(s\) would be used for RL.*inst-001"):
        sp.assert_disjoint(ds[1:], path, "RL")
    with pytest.raises(sp.FrozenLeakError, match="retrieval index"):
        sp.assert_disjoint(["inst-000"], {"inst-000"}, "retrieval index")
    renamed = copy.deepcopy(ds[0])
    renamed.instance_id = "renamed-copy"
    with pytest.raises(sp.FrozenLeakError, match="content of a frozen instance"):
        sp.assert_disjoint([renamed], path, "teacher hints")


# ---------------------------------------------------------------- eval set
def test_build_eval_set_filters(seed):
    ds = make(seed, 6)
    ds[0].metadata["flaky"] = True
    del ds[1].metadata["flaky"]
    ds[2].metadata["leakage"] = ["issue names FAIL_TO_PASS test: x"]
    ds[3].metadata["leakage"] = ["trajectory[0] view_file reads hidden test file"]  # ok for eval
    ds[4].metadata["issue_sufficient"] = False
    ds[5].metadata["issue_sufficient"] = True
    assert ids(sp.build_eval_set(ds)) == ["inst-003", "inst-004", "inst-005"]
    assert ids(sp.build_eval_set(ds, require_sufficient=True)) == ["inst-005"]
    assert len(sp.build_eval_set(ds, require_non_flaky=False, require_no_leakage=False)) == 6
    assert sp.eval_exclusion_reason(ds[1]).startswith("not validated")
    # leakage computed on the fly when metadata lacks it
    d = copy.deepcopy(ds[5])
    del d.metadata["leakage"]
    d.problem_statement += "\n" + d.FAIL_TO_PASS[0].split("::")[-1]
    assert sp.eval_exclusion_reason(d) == "test leakage in issue"


def test_sample_human_review(seed):
    ds = make(seed, 12)
    md = sp.sample_human_review(ds, 5, seed=1)
    assert md == sp.sample_human_review(ds, 5, seed=1)
    assert sum(l.startswith("## ") for l in md.splitlines()) == 5 and md.count("- [ ] ") == 15
    for s in ("solvable as written", "not the gold implementation", "Another correct fix", "```diff"):
        assert s in md
    picked = [l for l in md.splitlines() if l.startswith("## ")]
    types = {ds[int(l.split("inst-")[1][:3])].metadata["bug_type"] for l in picked}
    assert types == set(BUG_TYPES)  # stratified round-robin
    assert sp.sample_human_review(ds, 50).count("- [ ] ") == 36


# ---------------------------------------------------------------- CLI
def _parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sp.register(sub)
    return p


def test_cli_split_and_eval_set(seed, tmp_path):
    ds = make(seed, 20)
    for i, d in enumerate(ds):
        d.metadata["created_at"] = f"2024-02-{i + 1:02d}"
    ds[3].metadata["cluster_id"] = ds[4].metadata["cluster_id"] = "cluster:inst-003"
    src = tmp_path / "in.jsonl"
    save_demos(ds, src)
    out = tmp_path / "splits"
    p = _parser()
    a = p.parse_args(["split", str(src), "--out-dir", str(out), "--seed", "2", "--ratios", "0.7,0.15,0.15"])
    assert a.fn(a) == 0
    parts = {k: load_demos(out / f"{k}.jsonl") for k in sp.SPLITS}
    assert sum(len(v) for v in parts.values()) == 20 and sp.check_split_integrity(parts) == []
    assert sp.load_frozen_ids(out / "frozen_test.manifest.json") == set(ids(parts["test"]))
    report = (out / "split_report.md").read_text()
    assert "Integrity" in report and "OK" in report and "In-repo time split" in report
    assert (out / "in_repo_test.jsonl").exists()
    # re-running is idempotent; a different split would refuse to overwrite the frozen set
    assert a.fn(a) == 0
    b = p.parse_args(["split", str(src), "--out-dir", str(out), "--seed", "99"])
    if set(ids(sp.split_demos(ds, seed=99)["test"])) != set(ids(parts["test"])):
        assert b.fn(b) == 1
    # honoring the existing frozen manifest keeps the test set
    c = p.parse_args(["split", str(src), "--out-dir", str(out), "--seed", "99",
                      "--frozen", str(out / "frozen_test.manifest.json")])
    assert c.fn(c) == 0
    assert set(ids(parts["test"])) <= set(ids(load_demos(out / "test.jsonl")))

    ds[0].metadata["flaky"] = True
    save_demos(ds, src)
    ev, rv = tmp_path / "eval.jsonl", tmp_path / "review.md"
    e = p.parse_args(["eval-set", str(src), "--out", str(ev), "--human-review", "3", "--review-out", str(rv)])
    assert e.fn(e) == 0
    assert len(load_demos(ev)) == 19 and rv.read_text().count("- [ ] ") == 9
