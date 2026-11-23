"""Run database: every evaluation run, its per-sample results and full episodes, in one SQLite file.

  runs      one row per `repo-fix eval` (or import): model, scaffold config, dataset, git commit,
            sampling settings, status, summary, token usage, cost
  results   one row per (run, instance, sample): resolved, steps, grading detail, errors, timing
  episodes  one row per (run, instance, sample): trajectory + patch + metadata (zlib-compressed
            JSON), the rollout format distill / prefs / onpolicy / difficulty-weights consume
  instances one row per demo seen: difficulty, repo, bug type, content hash

  repo-fix db runs                                   # list runs, newest first
  repo-fix db show <run>                             # summary + per-difficulty accuracy
  repo-fix db compare <run-a> <run-b>                # per-instance flips between two runs
  repo-fix db import runs/baseline/student.jsonl --model ... --label ...
  repo-fix db export <run> --what results|episodes --out file.jsonl

`eval` writes here automatically (default runs/repofix.sqlite, or $REPOFIX_DB; --no-db to skip).
Run ids can be abbreviated to any unique prefix, or "latest".
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PATH = "runs/repofix.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    finished_at  REAL,
    status       TEXT NOT NULL,            -- running | done | failed | imported
    label        TEXT,
    kind         TEXT,                     -- student | teacher
    model        TEXT,                     -- resolved model id
    dataset      TEXT,
    dataset_hash TEXT,
    n_instances  INTEGER,
    samples      INTEGER,
    best_of      INTEGER,
    temperature  REAL,
    max_steps    INTEGER,
    sandbox      TEXT,
    scaffold     TEXT,                     -- JSON: serving / agent options + system prompt hash
    git_commit   TEXT,
    git_dirty    INTEGER,
    argv         TEXT,                     -- JSON
    summary      TEXT,                     -- JSON (evaluation.summarize)
    usage        TEXT,                     -- JSON
    cost_usd     REAL,
    accuracy     REAL
);
CREATE TABLE IF NOT EXISTS results (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    instance_id TEXT NOT NULL,
    sample      INTEGER NOT NULL DEFAULT 0,
    difficulty  TEXT,
    resolved    INTEGER,
    steps       INTEGER,
    submitted   INTEGER,
    error       TEXT,
    patch_lines INTEGER,
    f2p_passed  INTEGER,
    f2p_total   INTEGER,
    p2p_broken  INTEGER,
    seconds     REAL,
    extra       TEXT,                      -- JSON: any other fields (best-of oracle, grade_error, ...)
    PRIMARY KEY (run_id, instance_id, sample)
);
CREATE TABLE IF NOT EXISTS episodes (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    instance_id TEXT NOT NULL,
    sample      INTEGER NOT NULL DEFAULT 0,
    resolved    INTEGER,
    num_steps   INTEGER,
    demo        BLOB NOT NULL,             -- zlib(JSON of the episode Demo)
    PRIMARY KEY (run_id, instance_id, sample)
);
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    difficulty  TEXT,
    repo        TEXT,
    bug_type    TEXT,
    content_hash TEXT,
    first_seen  REAL
);
CREATE INDEX IF NOT EXISTS results_instance ON results(instance_id);
"""

RESULT_COLS = ("difficulty", "resolved", "steps", "submitted", "error", "patch_lines", "f2p_passed", "f2p_total",
               "p2p_broken", "seconds")


def db_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("REPOFIX_DB") or DEFAULT_PATH)


def git_state(cwd: str | Path = ".") -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def file_hash(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _j(x: Any) -> str | None:
    return None if x is None else json.dumps(x, ensure_ascii=False, default=str)


class RunDB:
    """Thread-safe (one connection, one lock); writes commit immediately so a crash keeps progress."""

    def __init__(self, path: str | Path | None = None):
        self.path = db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _exec(self, sql: str, args: Iterable = ()) -> None:
        with self.lock:
            self.conn.execute(sql, tuple(args))
            self.conn.commit()

    def _rows(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    # ---- writes ---------------------------------------------------------------------------------
    def start_run(self, *, model: str, kind: str, dataset: str | None = None, label: str | None = None,
                  n_instances: int | None = None, samples: int = 1, best_of: int = 1,
                  temperature: float | None = None, max_steps: int | None = None, sandbox: str | None = None,
                  scaffold: dict | None = None, argv: list[str] | None = None, status: str = "running",
                  created_at: float | None = None, run_id: str | None = None, record_git: bool = True) -> str:
        created = created_at or time.time()
        rid = run_id or f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created))}-{uuid.uuid4().hex[:6]}"
        commit, dirty = git_state() if record_git else (None, None)
        self._exec("INSERT INTO runs (run_id, created_at, status, label, kind, model, dataset, dataset_hash, "
                   "n_instances, samples, best_of, temperature, max_steps, sandbox, scaffold, git_commit, git_dirty, "
                   "argv) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (rid, created, status, label, kind, model, dataset, file_hash(dataset) if dataset else None,
                    n_instances, samples, best_of, temperature, max_steps, sandbox, _j(scaffold), commit,
                    None if dirty is None else int(dirty), _j(argv)))
        return rid

    def add_result(self, run_id: str, rec: dict) -> None:
        extra = {k: v for k, v in rec.items() if k not in RESULT_COLS + ("instance_id", "sample")}
        vals = [rec.get(c) for c in RESULT_COLS]
        for i, c in enumerate(RESULT_COLS):   # bools -> ints
            if isinstance(vals[i], bool):
                vals[i] = int(vals[i])
        self._exec(f"INSERT OR REPLACE INTO results (run_id, instance_id, sample, {', '.join(RESULT_COLS)}, extra) "
                   f"VALUES (?,?,?,{','.join('?' * len(RESULT_COLS))},?)",
                   (run_id, rec["instance_id"], int(rec.get("sample", 0)), *vals, _j(extra) if extra else None))

    def add_episode(self, run_id: str, demo) -> None:
        d = demo.to_dict() if hasattr(demo, "to_dict") else dict(demo)
        meta = d.get("metadata") or {}
        iid = meta.get("parent_id") or d["instance_id"]
        blob = zlib.compress(json.dumps(d, ensure_ascii=False).encode(), 6)
        self._exec("INSERT OR REPLACE INTO episodes (run_id, instance_id, sample, resolved, num_steps, demo) "
                   "VALUES (?,?,?,?,?,?)",
                   (run_id, iid, int(meta.get("sample_idx", 0)), None if meta.get("resolved") is None
                    else int(bool(meta["resolved"])), len(d.get("trajectory") or []), blob))

    def upsert_instances(self, demos: Iterable) -> None:
        now = time.time()
        rows = []
        for d in demos:
            m = d.metadata or {}
            h = hashlib.sha256(json.dumps([d.problem_statement, d.patch, d.test_patch, d.base_commit],
                                          default=str).encode()).hexdigest()[:16]
            rows.append((d.instance_id, m.get("difficulty"), d.repo, m.get("bug_type"), h, now))
        with self.lock:
            self.conn.executemany(
                "INSERT INTO instances (instance_id, difficulty, repo, bug_type, content_hash, first_seen) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(instance_id) DO UPDATE SET difficulty=excluded.difficulty, "
                "repo=excluded.repo, bug_type=excluded.bug_type, content_hash=excluded.content_hash", rows)
            self.conn.commit()

    def finish_run(self, run_id: str, summary: dict | None = None, usage: dict | None = None,
                   status: str = "done") -> None:
        s = summary or {}
        self._exec("UPDATE runs SET finished_at=?, status=?, summary=?, usage=?, cost_usd=?, accuracy=? WHERE run_id=?",
                   (time.time(), status, _j(summary), _j(usage), s.get("est_cost_usd"), s.get("accuracy"), run_id))

    def delete_run(self, run_id: str) -> None:
        self._exec("DELETE FROM runs WHERE run_id=?", (run_id,))

    # ---- reads ----------------------------------------------------------------------------------
    def resolve(self, ref: str) -> str:
        if ref == "latest":
            rows = self._rows("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1")
        else:
            rows = self._rows("SELECT run_id FROM runs WHERE run_id = ? OR run_id LIKE ? OR label = ? "
                              "ORDER BY created_at DESC", (ref, ref + "%", ref))
            exact = [r for r in rows if r["run_id"] == ref]
            rows = exact or rows
        ids = list(dict.fromkeys(r["run_id"] for r in rows))
        if not ids:
            raise KeyError(f"no run matches {ref!r}")
        if len(ids) > 1:
            raise KeyError(f"{ref!r} is ambiguous: {', '.join(ids[:5])}")
        return ids[0]

    def run(self, ref: str) -> dict:
        r = dict(self._rows("SELECT * FROM runs WHERE run_id=?", (self.resolve(ref),))[0])
        for k in ("scaffold", "argv", "summary", "usage"):
            r[k] = json.loads(r[k]) if r[k] else None
        return r

    def runs(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self._rows(
            "SELECT r.run_id, r.created_at, r.status, r.label, r.kind, r.model, r.samples, r.best_of, r.accuracy, "
            "r.git_commit, r.git_dirty, r.n_instances, (SELECT COUNT(*) FROM results x WHERE x.run_id=r.run_id) AS done,"
            " (SELECT SUM(resolved) FROM results x WHERE x.run_id=r.run_id) AS solved "
            "FROM runs r ORDER BY r.created_at DESC LIMIT ?", (limit,))]

    def results(self, ref: str) -> list[dict]:
        out = []
        for r in self._rows("SELECT * FROM results WHERE run_id=? ORDER BY instance_id, sample", (self.resolve(ref),)):
            d = {k: r[k] for k in r.keys() if k not in ("run_id", "extra")}
            for k in ("resolved", "submitted"):
                d[k] = None if d[k] is None else bool(d[k])
            d.update(json.loads(r["extra"]) if r["extra"] else {})
            out.append(d)
        return out

    def episodes(self, ref: str) -> list[dict]:
        return [json.loads(zlib.decompress(r["demo"])) for r in self._rows(
            "SELECT demo FROM episodes WHERE run_id=? ORDER BY instance_id, sample", (self.resolve(ref),))]

    def done_keys(self, ref: str) -> set[tuple[str, int]]:
        return {(r["instance_id"], r["sample"]) for r in self._rows(
            "SELECT instance_id, sample FROM results WHERE run_id=? AND error IS NULL", (self.resolve(ref),))}


# ---- analysis -----------------------------------------------------------------------------------------
def per_instance(results: list[dict]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in results:
        if r.get("error"):
            continue
        out.setdefault(r["instance_id"], []).append(bool(r.get("resolved")))
    return out


def compare(a: list[dict], b: list[dict]) -> dict:
    """Instances in both runs; an instance counts as solved when its mean pass rate is >= 0.5."""
    pa, pb = per_instance(a), per_instance(b)
    common = sorted(set(pa) & set(pb))
    rate = lambda v: sum(v) / len(v)  # noqa: E731
    cats: dict[str, list[str]] = {"both": [], "a_only": [], "b_only": [], "neither": []}
    for i in common:
        x, y = rate(pa[i]) >= 0.5, rate(pb[i]) >= 0.5
        cats["both" if x and y else "a_only" if x else "b_only" if y else "neither"].append(i)
    mean = lambda p: sum(rate(p[i]) for i in common) / len(common) if common else None  # noqa: E731
    return {"common": len(common), "a_mean": mean(pa), "b_mean": mean(pb), **cats,
            "only_in_a": sorted(set(pa) - set(pb)), "only_in_b": sorted(set(pb) - set(pa))}


# ---- CLI --------------------------------------------------------------------------------------------
def _fmt_time(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def cmd_runs(db: RunDB, a) -> int:
    rows = db.runs(a.limit)
    if not rows:
        print(f"no runs in {db.path}")
        return 0
    print(f"{'run':24} {'when':16} {'status':8} {'kind':7} {'acc':>6} {'done':>9}  {'commit':8} label / model")
    for r in rows:
        acc = r["accuracy"] if r["accuracy"] is not None else (r["solved"] or 0) / r["done"] if r["done"] else None
        tot = (r["n_instances"] or 0) * (r["samples"] or 1)
        done = f"{r['done']}/{tot}" if tot else str(r["done"])
        commit = (r["git_commit"] or "-")[:7] + ("*" if r["git_dirty"] else "")
        print(f"{r['run_id']:24} {_fmt_time(r['created_at']):16} {r['status']:8} {r['kind'] or '-':7} "
              f"{f'{acc:.1%}' if acc is not None else '-':>6} {done:>9}  {commit:8} {r['label'] or r['model']}")
    return 0


def cmd_show(db: RunDB, a) -> int:
    from .evaluation import format_summary, summarize

    r = db.run(a.run)
    res = db.results(r["run_id"])
    print(f"run {r['run_id']}  [{r['status']}]  {r['label'] or ''}")
    print(f"  model {r['model']} ({r['kind']}), dataset {r['dataset']} ({r['dataset_hash']}), "
          f"samples {r['samples']}, best_of {r['best_of']}, temperature {r['temperature']}")
    print(f"  commit {r['git_commit']}{' (dirty)' if r['git_dirty'] else ''}, started {_fmt_time(r['created_at'])}, "
          f"finished {_fmt_time(r['finished_at'])}")
    if r["scaffold"]:
        print(f"  scaffold {json.dumps(r['scaffold'])}")
    errs = Counter((x.get("error") or "").split(":")[0] for x in res if x.get("error"))
    if errs:
        print(f"  errors {dict(errs)}")
    if res:
        print(format_summary(summarize(r["model"] or "?", res, (r["usage"] or None))))
    return 0


def cmd_compare(db: RunDB, a) -> int:
    ra, rb = db.run(a.a), db.run(a.b)
    c = compare(db.results(ra["run_id"]), db.results(rb["run_id"]))
    name = lambda r: r["label"] or r["run_id"]  # noqa: E731
    print(f"A = {name(ra)} ({ra['model']})\nB = {name(rb)} ({rb['model']})")
    if not c["common"]:
        print("no instances in common")
        return 1
    print(f"{c['common']} instances in common: A {c['a_mean']:.1%}, B {c['b_mean']:.1%} "
          f"({(c['b_mean'] - c['a_mean']) * 100:+.1f} pts)")
    print(f"  both {len(c['both'])}, neither {len(c['neither'])}, A only {len(c['a_only'])}, B only {len(c['b_only'])}")
    for k, title in (("b_only", "fixed in B (A failed)"), ("a_only", "broken in B (A solved)")):
        if c[k]:
            print(f"  {title}: {', '.join(c[k])}")
    return 0


def cmd_import(db: RunDB, a) -> int:
    recs = [json.loads(ln) for ln in Path(a.results).read_text().splitlines() if ln.strip()]
    summary_p = Path(a.results).with_suffix(".summary.json")
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else None
    model = a.model or (summary or {}).get("model") or "unknown"
    kind = a.kind or ("teacher" if model.startswith(("claude-", "gpt-")) else "student")
    samples = max((int(r.get("sample", 0)) for r in recs), default=0) + 1
    rid = db.start_run(model=model, kind=kind, dataset=a.dataset, label=a.label or Path(a.results).stem,
                       n_instances=len({r["instance_id"] for r in recs}), samples=samples, status="imported",
                       created_at=Path(a.results).stat().st_mtime, argv=["import", a.results],
                       record_git=False)   # the code that produced an imported file is unknown
    for r in recs:
        db.add_result(rid, r)
    if a.rollouts:
        from .schema import load_demos

        for d in load_demos(a.rollouts):
            db.add_episode(rid, d)
    db.finish_run(rid, summary, (summary or {}).get("usage"), status="imported")
    print(f"imported {len(recs)} results as run {rid}")
    return 0


def cmd_export(db: RunDB, a) -> int:
    rows = db.results(a.run) if a.what == "results" else db.episodes(a.run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} {a.what} to {a.out}")
    return 0


def cmd_delete(db: RunDB, a) -> int:
    rid = db.resolve(a.run)
    if not a.yes:
        print(f"would delete run {rid} with its results and episodes; re-run with --yes")
        return 0
    db.delete_run(rid)
    print(f"deleted run {rid}")
    return 0


def cmd_db(a) -> int:
    db = RunDB(a.db)
    try:
        return {"runs": cmd_runs, "show": cmd_show, "compare": cmd_compare, "import": cmd_import,
                "export": cmd_export, "delete": cmd_delete}[a.db_cmd](db, a)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    finally:
        db.close()


def register(subparsers) -> None:
    p = subparsers.add_parser("db", help="run database: list / show / compare / import / export eval runs")
    p.add_argument("--db", default=None, help=f"SQLite path (default $REPOFIX_DB or {DEFAULT_PATH})")
    sub = p.add_subparsers(dest="db_cmd", required=True)
    r = sub.add_parser("runs", help="list runs, newest first")
    r.add_argument("--limit", type=int, default=50)
    s = sub.add_parser("show", help="one run's settings and summary")
    s.add_argument("run", help="run id, unique prefix, label, or 'latest'")
    c = sub.add_parser("compare", help="per-instance differences between two runs")
    c.add_argument("a")
    c.add_argument("b")
    i = sub.add_parser("import", help="import an eval results JSONL (+ .summary.json next to it)")
    i.add_argument("results")
    i.add_argument("--model", default=None)
    i.add_argument("--kind", choices=["student", "teacher"], default=None)
    i.add_argument("--label", default=None)
    i.add_argument("--dataset", default=None)
    i.add_argument("--rollouts", default=None, help="saved rollouts JSONL for the same run")
    e = sub.add_parser("export", help="write a run's results or episodes as JSONL (pipeline input)")
    e.add_argument("run")
    e.add_argument("--what", choices=["results", "episodes"], default="results")
    e.add_argument("--out", required=True)
    d = sub.add_parser("delete", help="delete a run (dry run unless --yes)")
    d.add_argument("run")
    d.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_db)
