"""Run database (rundb.py) and its eval integration. Offline: scripted policy, local sandbox."""
import json
from pathlib import Path

import pytest

from repo_fix_agent import cli
from repo_fix_agent import rundb as R
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import load_demos

ROOT = Path(__file__).resolve().parents[1]
IDS = ["seed-001-pagination-offset", "seed-003-leap-year-century"]


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("REPOFIX_SANDBOX", "local")


def _res(iid, ok, sample=0, **kw):
    return {"instance_id": iid, "sample": sample, "difficulty": "easy", "resolved": ok, "steps": 3,
            "submitted": True, "error": None, **kw}


def test_write_read_compare(tmp_path):
    db = R.RunDB(tmp_path / "x.sqlite")
    a = db.start_run(model="m", kind="student", label="A", n_instances=3)
    for iid, ok in (("i1", True), ("i2", False), ("i3", True)):
        db.add_result(a, _res(iid, ok, oracle_resolved=True))
    db.finish_run(a, {"accuracy": 2 / 3})
    b = db.start_run(model="m", kind="student", label="B", n_instances=3, samples=2)
    for iid, oks in (("i1", [1, 1]), ("i2", [1, 0]), ("i3", [0, 0])):
        for k, ok in enumerate(oks):
            db.add_result(b, _res(iid, bool(ok), k))
    db.add_result(b, _res("i4", False, error="APIError: boom"))
    assert db.resolve("A") == a and db.resolve(b[:-2]) == b and db.resolve("latest") == b
    with pytest.raises(KeyError):
        db.resolve("nope")
    r = db.results(a)
    assert r[0]["resolved"] is True and r[0]["oracle_resolved"] is True          # extra fields round-trip
    c = R.compare(db.results(a), db.results(b))
    assert c["common"] == 3 and c["b_only"] == ["i2"] and c["a_only"] == ["i3"] and c["both"] == ["i1"]
    assert c["a_mean"] == pytest.approx(2 / 3) and c["b_mean"] == pytest.approx(0.5)
    assert db.done_keys(b) == {("i1", 0), ("i1", 1), ("i2", 0), ("i2", 1), ("i3", 0), ("i3", 1)}   # errors not done
    assert db.run(a)["accuracy"] == pytest.approx(2 / 3) and db.run(a)["git_commit"]
    db.delete_run(a)
    assert not db._rows("SELECT * FROM results WHERE run_id=?", (a,))                 # cascade
    db.close()


def test_eval_records_run_episodes_and_resumes(tmp_path, monkeypatch, capsys):
    demos = {d.instance_id: d for d in load_demos(ROOT / "data/demos/seed.jsonl")}
    script = lambda d: ScriptedPolicy([(s.thought, s.action) for s in d.trajectory])  # noqa: E731
    current = {}

    class ByTask:   # replays the gold trajectory of whichever instance the agent is given
        def reset(self, system, task):
            iid = next(i for i in IDS if demos[i].problem_statement in (task if isinstance(task, str) else str(task)))
            current["p"] = script(demos[iid])
            current["p"].reset(system, task)

        def act(self, obs):
            return current["p"].act(obs)

    monkeypatch.setattr(cli, "make_policy", lambda m, t=None: ByTask())
    dbp, out = tmp_path / "runs.sqlite", tmp_path / "eval.jsonl"
    argv = ["eval", str(ROOT / "data/demos/seed.jsonl"), "--model", "claude-test", "--ids", ",".join(IDS),
            "--root", str(ROOT), "--db", str(dbp), "--label", "gold replay", "--out", str(out)]
    assert cli.main(argv) == 0
    db = R.RunDB(dbp)
    run = db.run("gold replay")
    assert run["status"] == "done" and run["accuracy"] == 1.0 and run["n_instances"] == 2
    assert run["scaffold"]["policy"] == "ByTask" and run["scaffold"]["system_prompt_sha"]
    assert {r["instance_id"] for r in db.results(run["run_id"])} == set(IDS)
    eps = db.episodes(run["run_id"])
    assert len(eps) == 2 and all(e["trajectory"] and e["patch"] for e in eps)
    assert {i["instance_id"] for i in db._rows("SELECT instance_id FROM instances")} == set(IDS)

    # resume: nothing left to do -> no new episodes, summary still covers both
    monkeypatch.setattr(cli, "make_policy", lambda m, t=None: ByTask())
    assert cli.main(argv[:-4] + ["--resume", run["run_id"], "--db", str(dbp)]) == 0
    run2 = db.run(run["run_id"])
    assert run2["status"] == "done" and run2["summary"]["n"] == 2
    assert len(db.runs()) == 1

    # export feeds the pipeline (difficulty-weights reads results or episodes)
    exp = tmp_path / "eps.jsonl"
    assert cli.main(["db", "--db", str(dbp), "export", run["run_id"], "--what", "episodes", "--out", str(exp)]) == 0
    from repo_fix_agent.pipeline.difficulty import pass_counts
    assert pass_counts([json.loads(x) for x in exp.read_text().splitlines()]) == {i: (1, 1) for i in IDS}
    db.close()


def test_no_db_and_import(tmp_path, monkeypatch):
    res = tmp_path / "old.jsonl"
    res.write_text("".join(json.dumps(_res(f"i{k}", k % 2 == 0)) + "\n" for k in range(4)))
    res.with_suffix(".summary.json").write_text(json.dumps({"model": "claude-x", "accuracy": 0.5}))
    dbp = tmp_path / "d.sqlite"
    assert cli.main(["db", "--db", str(dbp), "import", str(res), "--label", "old"]) == 0
    db = R.RunDB(dbp)
    r = db.run("old")
    assert r["kind"] == "teacher" and r["status"] == "imported" and r["git_commit"] is None and r["accuracy"] == 0.5
    assert cli.main(["db", "--db", str(dbp), "delete", "old"]) == 0 and db.run("old")     # dry run keeps it
    assert cli.main(["db", "--db", str(dbp), "show", "nope"]) == 2
    db.close()
