import argparse
import copy
import json
from pathlib import Path

import pytest

from repo_fix_agent.pipeline import retrieval as R
from repo_fix_agent.pipeline.verifier import ExcludedInstanceError
from repo_fix_agent.schema import load_demos, save_demos

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/demos/seed.jsonl"
PAGINATION = "seed-001-pagination-offset"
PARAPHRASE = ("When I request page 1 with 10 items per page I get back the second batch (10..19) of records. "
              "Paging looks shifted by one, and the final page can come back empty.")


def _demos():
    return [copy.deepcopy(d) for d in load_demos(SEED)]


def _by_id(demos, iid):
    return next(d for d in demos if d.instance_id == iid)


def test_tokenize_and_summary():
    toks = R.tokenize("Pages paging per_page totalPages the")
    assert toks.count("pag") >= 3 and "the" not in toks
    d = _by_id(_demos(), PAGINATION)
    s = R.summarize_patch(d.patch)
    assert "- start = page * per_page" in s and "+ start = (page - 1) * per_page" in s


def test_paraphrased_issue_finds_right_neighbor():
    demos = _demos()
    idx = R.RetrievalIndex.build(demos)
    hits = idx.query(PARAPHRASE, k=2)
    assert hits[0].entry["instance_id"] == PAGINATION
    assert hits[0].entry["files_changed"] == ["webapp/pagination.py"]
    assert hits[0].cosine > hits[1].cosine


def test_same_repo_boost():
    demos = _demos()
    idx = R.RetrievalIndex.build(demos)
    other = next(d for d in demos if d.instance_id != PAGINATION)
    assert idx.query(PARAPHRASE, repo=other.repo, same_repo_boost=0.0)[0].entry["instance_id"] == PAGINATION
    top = idx.query(PARAPHRASE, repo=other.repo, same_repo_boost=1.0)[0]
    assert top.entry["instance_id"] == other.instance_id and top.same_repo
    # basename / trailing slash / .git differences still count as the same repo
    assert idx.query("x", repo="/elsewhere/" + Path(other.repo).name + ".git/", k=1)[0].same_repo


def test_exclude_ids_refused_and_unresolved_skipped():
    demos = _demos()
    with pytest.raises(ExcludedInstanceError):
        R.RetrievalIndex.build(demos, exclude_ids={PAGINATION})
    rollout = copy.deepcopy(demos[0])
    rollout.instance_id = "r1"
    rollout.metadata = {"parent_id": "frozen-1", "resolved": True}
    with pytest.raises(ExcludedInstanceError):
        R.RetrievalIndex.build([rollout], exclude_ids={"frozen-1"})
    bad = copy.deepcopy(demos[0])
    bad.metadata = {"resolved": False}
    assert len(R.RetrievalIndex.build([bad])) == 0


def test_save_load_roundtrip(tmp_path):
    idx = R.RetrievalIndex.build(_demos())
    p = tmp_path / "index.json"
    idx.save(p)
    idx2 = R.RetrievalIndex.load(p)
    a = [(h.entry["instance_id"], round(h.score, 9)) for h in idx.query(PARAPHRASE, k=3)]
    b = [(h.entry["instance_id"], round(h.score, 9)) for h in idx2.query(PARAPHRASE, k=3)]
    assert a == b


def test_format_exemplars_and_augment():
    idx = R.RetrievalIndex.build(_demos())
    hits = idx.query(PARAPHRASE, k=2)
    block = R.format_exemplars(hits, max_chars=1500)
    assert len(block) <= 1500 and "webapp/pagination.py" in block and "Patch summary" in block
    assert "[1]" in block and "Submitted." not in block  # no trajectories
    short = R.format_exemplars(hits, max_chars=120)
    assert len(short) <= 120 and short.endswith("...")
    assert R.format_exemplars([]) == ""
    task = R.augment_task(PARAPHRASE, idx)
    assert task.startswith(PARAPHRASE) and R.EXEMPLAR_HEADER in task
    assert R.augment_task("zzqx", idx) == "zzqx"


def _parser():
    p = argparse.ArgumentParser()
    R.register(p.add_subparsers(dest="cmd", required=True))
    return p


def test_cli(tmp_path, capsys):
    inp = tmp_path / "train.jsonl"
    save_demos(_demos(), inp)
    out = tmp_path / "index.json"
    a = _parser().parse_args(["retrieval-index", str(inp), "--out", str(out)])
    assert a.fn(a) == 0
    issue = tmp_path / "issue.md"
    issue.write_text(PARAPHRASE)
    a = _parser().parse_args(["retrieve", "--index", str(out), "--issue-file", str(issue), "--k", "1", "--json"])
    capsys.readouterr()
    assert a.fn(a) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["instance_id"] == PAGINATION
    a = _parser().parse_args(["retrieve", "--index", str(out), "--issue", PARAPHRASE])
    assert a.fn(a) == 0 and "pagination" in capsys.readouterr().out

    excl = tmp_path / "frozen.json"
    excl.write_text(json.dumps({"instances": [PAGINATION]}))
    a = _parser().parse_args(["retrieval-index", str(inp), "--out", str(out), "--exclude-ids", str(excl)])
    assert a.fn(a) == 2
