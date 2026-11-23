"""Backtest dashboard (dashboard.py). Offline."""
import json
import re

import pytest

from repo_fix_agent import dashboard as DB
from repo_fix_agent.rundb import RunDB


def _r(iid, ok, diff="easy", **kw):
    return {"instance_id": iid, "sample": 0, "difficulty": diff, "resolved": ok, "steps": 10, "submitted": True,
            "error": None, "f2p_passed": 1 if ok else 0, "f2p_total": 1, "p2p_broken": 0, **kw}


@pytest.fixture
def db(tmp_path):
    d = RunDB(tmp_path / "x.sqlite")
    a = d.start_run(model="m", kind="student", label="v1", n_instances=3, max_steps=40, created_at=1000)
    for r in (_r("i1", True), _r("i2", True, "hard"), _r("i3", False, "medium")):
        d.add_result(a, r)
    d.finish_run(a, {"accuracy": 2 / 3}, {"calls": 10})
    t = d.start_run(model="claude", kind="teacher", label="teacher", n_instances=3, created_at=1500)
    for r in (_r("i1", True), _r("i2", True, "hard"), _r("i3", True, "medium")):
        d.add_result(t, r)
    d.finish_run(t, {"accuracy": 1.0})
    b = d.start_run(model="m", kind="student", label="v2", n_instances=3, max_steps=40, created_at=2000)
    for r in (_r("i1", True), _r("i2", False, "hard", submitted=False, steps=40),
              _r("i3", True, "medium")):
        d.add_result(b, r)
    d.finish_run(b, {"accuracy": 2 / 3})
    running = d.start_run(model="m", kind="student", label="partial", n_instances=3, created_at=3000)
    d.add_result(running, _r("i1", True))
    yield d
    d.close()


def test_failure_modes():
    fm = lambda **kw: DB.failure_mode({**_r("x", False), **kw}, 40)  # noqa: E731
    assert DB.failure_mode(_r("x", True), 40) is None
    assert fm(error="APIError") == "error"
    assert fm(submitted=False, steps=40) == "ran out of steps" and fm(submitted=False, steps=12) == "gave up"
    assert fm(p2p_broken=1) == "broke a passing test"
    assert fm(f2p_passed=1, f2p_total=2) == "partial fix" and fm() == "wrong fix"


def test_build_data_baseline_flips_and_matrix(db):
    d = DB.build_data(db)
    labels = [r["label"] for r in d["runs"]]
    assert labels == ["v1", "teacher", "v2"]                      # in-progress run left out
    latest, prev = d["latest"], d["previous"]
    assert db.run(latest)["label"] == "v2" and db.run(prev)["label"] == "v1"   # same kind, not the teacher
    assert d["flips"]["fixed"] == ["i3"] and d["flips"]["broken"] == ["i2"]
    v2 = next(r for r in d["runs"] if r["label"] == "v2")
    assert v2["failure_modes"]["ran out of steps"] == 1 and v2["by_difficulty"]["hard"]["accuracy"] == 0
    ids = [i["id"] for i in d["matrix"]["instances"]]
    assert set(ids[:2]) == {"i2", "i3"} and ids[2] == "i1"        # changed instances first
    assert d["matrix"]["cells"][latest]["i2"]["mode"] == "ran out of steps"
    partial = DB.build_data(db, run="partial")                    # featured explicitly -> included
    assert db.run(partial["latest"])["label"] == "partial"
    assert db.run(DB.build_data(db, baseline="teacher")["previous"])["label"] == "teacher"


def test_render_is_self_contained(db, tmp_path):
    out = DB.write_dashboard(db, tmp_path / "d.html")
    html = out.read_text()
    assert "<title>Backtest: v2</title>" in html
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', html)   # no external resources
    payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload)["latest"] == db.resolve("v2")
    assert "prefers-color-scheme: dark" in html and ':root[data-theme="dark"]' in html


def test_script_injection_is_escaped(db, tmp_path):
    rid = db.start_run(model="m", kind="student", label="</script><script>alert(1)</script>", n_instances=1,
                       created_at=4000)
    db.add_result(rid, _r("i1", True))
    db.finish_run(rid, {"accuracy": 1.0})
    html = DB.render_html(DB.build_data(db))
    assert "</script><script>alert(1)" not in html


def test_cli(db, capsys, tmp_path):
    import argparse
    p = argparse.ArgumentParser()
    DB.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["dashboard", "--db", str(db.path), "--out", str(tmp_path / "o.html")])
    assert a.fn(a) == 0 and (tmp_path / "o.html").exists()
    empty = RunDB(tmp_path / "empty.sqlite"); empty.close()
    a = p.parse_args(["dashboard", "--db", str(tmp_path / "empty.sqlite"), "--out", str(tmp_path / "e.html")])
    assert a.fn(a) == 2 and "no runs" in capsys.readouterr().err
