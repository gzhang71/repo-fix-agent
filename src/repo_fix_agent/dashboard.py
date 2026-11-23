"""Backtest dashboard: one self-contained HTML page built from the run database.

  repo-fix dashboard                         # -> runs/dashboard.html (latest run vs the previous one)
  repo-fix dashboard --run <id> --baseline <id> --out path.html

`repo-fix eval` regenerates it after every run (--no-dashboard to skip). The page has no external
dependencies, so it opens from disk and can be attached or archived as-is. It shows:
  * headline tiles: latest accuracy, change vs the baseline run, steps, API calls, cost
  * accuracy across runs (student / teacher series)
  * accuracy by difficulty, baseline vs latest
  * why the latest run's failures failed (failure modes)
  * an instance x run matrix (pass rate per cell), changed instances first
  * the instances that flipped, and a table of every run
"""
from __future__ import annotations

import html
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .rundb import RunDB, compare

DEFAULT_OUT = "runs/dashboard.html"
DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}
FAILURE_MODES = ("error", "ran out of steps", "gave up", "broke a passing test", "partial fix", "wrong fix")


def failure_mode(r: dict, max_steps: int | None) -> str | None:
    if r.get("error"):
        return "error"
    if r.get("resolved"):
        return None
    if not r.get("submitted"):
        return "ran out of steps" if max_steps and (r.get("steps") or 0) >= max_steps else "gave up"
    if (r.get("p2p_broken") or 0) > 0:
        return "broke a passing test"
    if (r.get("f2p_passed") or 0) > 0:
        return "partial fix"
    return "wrong fix"


def run_stats(run: dict, results: list[dict]) -> dict:
    graded = [r for r in results if not r.get("error")]
    by_d: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in graded:
        d = r.get("difficulty") or "unknown"
        by_d[d][0] += bool(r.get("resolved"))
        by_d[d][1] += 1
    usage = run.get("usage") or {}
    summary = run.get("summary") or {}
    max_steps = run.get("max_steps") or (summary.get("max_steps")) or 40
    modes = Counter(m for m in (failure_mode(r, max_steps) for r in results) if m)
    solved = sum(bool(r.get("resolved")) for r in graded)
    return {
        "run_id": run["run_id"], "label": run.get("label") or run["run_id"], "kind": run.get("kind") or "?",
        "model": run.get("model"), "created_at": run.get("created_at"), "status": run.get("status"),
        "samples": run.get("samples") or 1, "n": len(graded), "errors": len(results) - len(graded),
        "solved": solved, "accuracy": solved / len(graded) if graded else None,
        "mean_steps": (sum(r.get("steps") or 0 for r in graded) / len(graded)) if graded else None,
        "calls": usage.get("calls"), "cost": run.get("cost_usd"),
        "commit": (run.get("git_commit") or "")[:7] or None, "dirty": bool(run.get("git_dirty")),
        "by_difficulty": {d: {"solved": v[0], "n": v[1], "accuracy": v[0] / v[1]}
                          for d, v in sorted(by_d.items(), key=lambda kv: DIFF_ORDER.get(kv[0], 9))},
        "failure_modes": {m: modes.get(m, 0) for m in FAILURE_MODES},
        "scaffold": run.get("scaffold"),
    }


def previous_comparable(runs: list[dict], latest: dict, results: dict[str, list[dict]]) -> dict | None:
    """The most recent earlier run of the same kind that shares at least half the instances."""
    mine = {r["instance_id"] for r in results[latest["run_id"]]}
    for r in sorted(runs, key=lambda x: x["created_at"] or 0, reverse=True):
        if r["run_id"] == latest["run_id"] or (r["created_at"] or 0) > (latest["created_at"] or 0):
            continue
        if r.get("kind") != latest.get("kind"):
            continue
        theirs = {x["instance_id"] for x in results[r["run_id"]]}
        if mine and len(mine & theirs) >= len(mine) / 2:
            return r
    return None


def build_data(db: RunDB, run: str | None = None, baseline: str | None = None, max_matrix_runs: int = 12) -> dict:
    runs = [db.run(r["run_id"]) for r in db.runs(limit=500)]
    featured = db.resolve(run) if run else None
    # unfinished runs would plot partial accuracy as if complete: only shown when explicitly featured
    runs = [r for r in runs if r["status"] in ("done", "imported") or r["run_id"] == featured]
    if not runs:
        raise ValueError("the run database has no runs yet")
    results = {r["run_id"]: db.results(r["run_id"]) for r in runs}
    runs = [r for r in runs if results[r["run_id"]]]
    runs.sort(key=lambda r: r["created_at"] or 0)
    latest = next(r for r in runs if r["run_id"] == db.resolve(run)) if run else runs[-1]
    prev = (next(r for r in runs if r["run_id"] == db.resolve(baseline)) if baseline
            else previous_comparable(runs, latest, results))
    stats = [run_stats(r, results[r["run_id"]]) for r in runs]

    shown = [r for r in runs if r["kind"] == latest["kind"] or r["run_id"] == (prev or {}).get("run_id")]
    shown = shown[-max_matrix_runs:]
    if latest not in shown:
        shown = shown[1:] + [latest]
    diff: dict[str, str] = {}
    cells: dict[str, dict[str, dict]] = {}
    for r in shown:
        per: dict[str, list] = defaultdict(list)
        for x in results[r["run_id"]]:
            per[x["instance_id"]].append(x)
            diff.setdefault(x["instance_id"], x.get("difficulty") or "unknown")
        cells[r["run_id"]] = {}
        for iid, xs in per.items():
            ok = [bool(x.get("resolved")) for x in xs if not x.get("error")]
            cells[r["run_id"]][iid] = {"p": sum(ok) / len(ok) if ok else None, "n": len(ok),
                                       "err": len(xs) - len(ok),
                                       "mode": failure_mode(xs[0], r.get("max_steps") or 40) if len(xs) == 1 else None,
                                       "steps": xs[0].get("steps") if len(xs) == 1 else None}

    def changed(iid: str) -> bool:
        vals = {round(cells[r["run_id"]][iid]["p"] or 0) for r in shown
                if iid in cells[r["run_id"]] and cells[r["run_id"]][iid]["p"] is not None}
        return len(vals) > 1

    instances = sorted(diff, key=lambda i: (not changed(i), DIFF_ORDER.get(diff[i], 9), i))
    flips = None
    if prev:
        c = compare(results[prev["run_id"]], results[latest["run_id"]])
        flips = {"fixed": c["b_only"], "broken": c["a_only"], "both": len(c["both"]), "neither": len(c["neither"]),
                 "common": c["common"], "prev_mean": c["a_mean"], "latest_mean": c["b_mean"]}
    return {
        "generated_at": time.time(), "db": str(db.path),
        "runs": stats, "latest": latest["run_id"], "previous": prev["run_id"] if prev else None,
        "matrix": {"runs": [r["run_id"] for r in shown],
                   "instances": [{"id": i, "difficulty": diff[i], "changed": changed(i)} for i in instances],
                   "cells": cells},
        "flips": flips,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    latest = next(r for r in data["runs"] if r["run_id"] == data["latest"])
    title = f"Backtest: {latest['label']}"
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


def write_dashboard(db: RunDB, out: str | Path = DEFAULT_OUT, run: str | None = None,
                    baseline: str | None = None) -> Path:
    data = build_data(db, run, baseline)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data))
    return out


def cmd_dashboard(a) -> int:
    import sys

    db = RunDB(a.db)
    try:
        out = write_dashboard(db, a.out, a.run, a.baseline)
    except (KeyError, ValueError) as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    finally:
        db.close()
    print(f"dashboard: {out.resolve()}")
    if a.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("dashboard", help="backtest dashboard (HTML) from the run database")
    p.add_argument("--db", default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--run", default=None, help="run to feature (default: the latest)")
    p.add_argument("--baseline", default=None, help="run to compare against (default: previous run of the same kind)")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.set_defaults(fn=cmd_dashboard)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834;
  --seq-0: #f0efec; --seq-1: #b7d3f6; --seq-2: #6da7ec; --seq-3: #2a78d6; --seq-4: #184f95;
  --prev: #86b6ef; --cur: #2a78d6;
  --good: #006300; --bad: #d03b3b; --warn: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
    --seq-0: #2c2c2a; --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #3987e5; --seq-4: #86b6ef;
    --prev: #184f95; --cur: #3987e5;
    --good: #0ca30c; --bad: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
  --seq-0: #2c2c2a; --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #3987e5; --seq-4: #86b6ef;
  --prev: #184f95; --cur: #3987e5;
  --good: #0ca30c; --bad: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 16px; }
h1 { font-size: 20px; margin: 0; font-weight: 650; }
h2 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
.sub { color: var(--ink-2); font-size: 13px; }
.muted { color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }
.tile, .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; }
.tile { padding: 14px 16px; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 26px; font-weight: 650; margin-top: 2px; }
.tile .d { font-size: 12px; margin-top: 2px; color: var(--ink-2); }
.up { color: var(--good); } .down { color: var(--bad); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.card { padding: 16px; margin-bottom: 16px; min-width: 0; }
.card .chart { margin-top: 10px; position: relative; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .lbl { fill: var(--ink-2); font-size: 11px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-2); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hl td { background: color-mix(in srgb, var(--series-1) 9%, transparent); }
.scroll { overflow-x: auto; }
.flips { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 8px; }
.flips ul { margin: 6px 0 0; padding-left: 18px; font-size: 12.5px; }
.tag { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px; border: 1px solid var(--ring); color: var(--ink-2); }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink);
       border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
       box-shadow: 0 4px 16px rgba(0,0,0,.12); max-width: 320px; display: none; }
#tip strong { font-weight: 650; }
#tip .row { display: flex; gap: 8px; justify-content: space-between; }
.matrix-wrap { overflow-x: auto; }
.toggle { font: inherit; font-size: 12px; color: var(--ink-2); background: none; border: 1px solid var(--ring);
          border-radius: 6px; padding: 3px 8px; cursor: pointer; }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1 id="title"></h1>
      <div class="sub" id="subtitle"></div>
    </div>
    <div class="sub" id="generated"></div>
  </header>
  <section class="tiles" id="tiles"></section>

  <div class="grid2">
    <section class="card"><h2>Accuracy across runs</h2>
      <div class="sub">Resolved share per run, in the order they ran; mean pass@1 for multi-sample runs. Runs under half this run's size are left out.</div>
      <div class="chart" id="trend"></div><div class="legend" id="trend-legend"></div></section>
    <section class="card"><h2>Accuracy by difficulty</h2>
      <div class="sub" id="diff-sub"></div>
      <div class="chart" id="bydiff"></div><div class="legend" id="diff-legend"></div></section>
  </div>

  <div class="grid2">
    <section class="card"><h2>Why the failures failed</h2>
      <div class="sub" id="modes-sub"></div>
      <div class="chart" id="modes"></div></section>
    <section class="card"><h2>What changed</h2>
      <div class="sub" id="flips-sub"></div>
      <div class="flips" id="flips"></div></section>
  </div>

  <section class="card"><h2>Instances by run</h2>
    <div class="sub">Each cell is one instance in one run: darker = solved more often. Instances whose result changed between the runs shown come first.
      <button class="toggle" id="only-changed" aria-pressed="false">Show changed only</button></div>
    <div class="legend" id="matrix-legend"></div>
    <div class="chart matrix-wrap" id="matrix"></div></section>

  <section class="card"><h2>All runs</h2>
    <div class="scroll"><table id="runs"></table></div></section>
</main>
<div id="tip" role="status"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const byId = Object.fromEntries(D.runs.map(r => [r.run_id, r]));
const L = byId[D.latest], P = D.previous ? byId[D.previous] : null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (x, d = 1) => x == null ? "–" : (x * 100).toFixed(d) + "%";
const fmtTime = t => t ? new Date(t * 1000).toLocaleString([], {year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}) : "–";
const el = (tag, attrs = {}, text) => { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
const svgEl = (tag, attrs = {}) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
const short = s => s.length > 46 ? s.slice(0, 44) + "…" : s;

// ---- tooltip (values lead, labels follow; built with textContent) ----
const tip = document.getElementById("tip");
function showTip(ev, title, rows) {
  tip.replaceChildren();
  tip.appendChild(el("div", {}, title)).style.marginBottom = rows.length ? "4px" : "0";
  for (const [k, v] of rows) { const r = el("div", {class: "row"}); r.appendChild(el("strong", {}, v)); r.appendChild(el("span", {class: "muted"}, k)); tip.appendChild(r); }
  tip.style.display = "block";
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.display = "none"; };

// ---- header + tiles ----
document.getElementById("title").textContent = "Backtest · " + L.label + (L.status === "running" ? " (in progress)" : L.status === "failed" ? " (failed)" : "");
document.getElementById("subtitle").textContent = `${L.model} · ${L.kind} · ${L.samples} sample(s) per instance · ${L.commit ? "commit " + L.commit + (L.dirty ? " (uncommitted changes)" : "") : "commit unknown"}`;
document.getElementById("generated").textContent = "Generated " + fmtTime(D.generated_at);
const tiles = document.getElementById("tiles");
function tile(k, v, d, cls) { const t = el("div", {class: "tile"}); t.appendChild(el("div", {class: "k"}, k));
  t.appendChild(el("div", {class: "v"}, v)); if (d) t.appendChild(el("div", {class: "d " + (cls || "")}, d)); tiles.appendChild(t); }
const delta = P && L.accuracy != null && P.accuracy != null ? (L.accuracy - P.accuracy) * 100 : null;
tile("Resolved", pct(L.accuracy), `${L.solved} of ${L.n}${L.errors ? ` · ${L.errors} errored` : ""}`);
tile("Change vs baseline", delta == null ? "–" : (delta > 0 ? "▲ +" : delta < 0 ? "▼ " : "") + delta.toFixed(1) + " pts",
     P ? "baseline: " + short(P.label) : "no comparable earlier run", delta > 0 ? "up" : delta < 0 ? "down" : "");
tile("Mean steps", L.mean_steps == null ? "–" : L.mean_steps.toFixed(1), P && P.mean_steps != null ? `baseline ${P.mean_steps.toFixed(1)}` : "");
tile("API calls", L.calls == null ? "–" : L.calls.toLocaleString(), P && P.calls != null ? `baseline ${P.calls.toLocaleString()}` : "live calls (cache replays excluded)");
tile("Est. cost", L.cost == null ? "–" : "$" + L.cost.toFixed(2), L.cost == null ? "no price for this model" : "");

// ---- chart helpers ----
function frame(container, height, m) {
  container.replaceChildren();
  const W = Math.max(280, container.clientWidth), H = height;
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img"});
  container.appendChild(svg);
  return {svg, W, H, x0: m.l, x1: W - m.r, y0: H - m.b, y1: m.t};
}
function yAxis(f, lo, hi, ticks, fmt) {
  const y = v => f.y0 - (v - lo) / (hi - lo) * (f.y0 - f.y1);
  for (const t of ticks) {
    f.svg.appendChild(svgEl("line", {x1: f.x0, x2: f.x1, y1: y(t), y2: y(t), stroke: css(t === lo ? "--axis" : "--grid"), "stroke-width": 1}));
    const tx = svgEl("text", {x: f.x0 - 6, y: y(t) + 3.5, "text-anchor": "end"}); tx.textContent = fmt(t); f.svg.appendChild(tx);
  }
  return y;
}

// ---- 1. accuracy across runs (line per kind) ----
function drawTrend() {
  const box = document.getElementById("trend");
  // runs much smaller than the featured one (pilots, --limit smoke runs) would read as full results
  const runs = D.runs.filter(r => r.accuracy != null && r.n >= L.n / 2);
  const f = frame(box, 240, {l: 40, r: 64, t: 12, b: 28});
  const lo = Math.max(0, Math.floor((Math.min(...runs.map(r => r.accuracy)) * 100 - 6) / 10) / 10), hi = 1;
  const ticks = []; for (let t = lo; t <= hi + 1e-9; t += 0.1) ticks.push(+t.toFixed(2));
  const y = yAxis(f, lo, hi, ticks, t => Math.round(t * 100) + "%");
  const x = i => runs.length === 1 ? (f.x0 + f.x1) / 2 : f.x0 + 12 + i * (f.x1 - f.x0 - 24) / (runs.length - 1);
  const num = r => D.runs.indexOf(r) + 1;   // same numbering as the runs table
  runs.forEach((r, i) => { const t = svgEl("text", {x: x(i), y: f.y0 + 16, "text-anchor": "middle"}); t.textContent = "#" + num(r); f.svg.appendChild(t); });
  const kinds = [...new Set(runs.map(r => r.kind))].sort((a, b) => (a === "student" ? -1 : b === "student" ? 1 : a.localeCompare(b)));
  const color = k => css(kinds.indexOf(k) === 0 ? "--series-1" : "--series-2");
  const legend = document.getElementById("trend-legend"); legend.replaceChildren();
  for (const k of kinds) {
    const pts = runs.map((r, i) => [r, i]).filter(([r]) => r.kind === k);
    if (pts.length > 1) f.svg.appendChild(svgEl("polyline", {points: pts.map(([r, i]) => `${x(i)},${y(r.accuracy)}`).join(" "),
      fill: "none", stroke: color(k), "stroke-width": 2, "stroke-linejoin": "round"}));
    for (const [r, i] of pts) {
      const isL = r.run_id === D.latest;
      f.svg.appendChild(svgEl("circle", {cx: x(i), cy: y(r.accuracy), r: isL ? 5.5 : 4, fill: color(k), stroke: css("--surface"), "stroke-width": 2}));
    }
    const [lr, li] = pts[pts.length - 1];
    const lab = svgEl("text", {x: x(li) + 9, y: y(lr.accuracy) + 4, class: "lbl"}); lab.textContent = k + " " + pct(lr.accuracy, 0); f.svg.appendChild(lab);
    const item = el("span"); const sw = el("i"); sw.style.background = color(k); item.append(sw, k); legend.appendChild(item);
  }
  // crosshair + one tooltip per run
  const hair = svgEl("line", {y1: f.y1, y2: f.y0, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden"}); f.svg.appendChild(hair);
  const hit = svgEl("rect", {x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, fill: "transparent"}); f.svg.appendChild(hit);
  hit.addEventListener("pointermove", ev => {
    const bx = f.svg.getBoundingClientRect().left; let best = 0, bd = 1e9;
    runs.forEach((r, i) => { const d = Math.abs(ev.clientX - bx - x(i)); if (d < bd) { bd = d; best = i; } });
    const r = runs[best]; hair.setAttribute("x1", x(best)); hair.setAttribute("x2", x(best)); hair.setAttribute("visibility", "visible");
    showTip(ev, `#${num(r)} · ${r.label}`, [["resolved", pct(r.accuracy)], ["kind", r.kind], ["mean steps", r.mean_steps == null ? "–" : r.mean_steps.toFixed(1)], ["run", r.run_id], ["when", fmtTime(r.created_at)]]);
  });
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTip(); });
}

// ---- 2. accuracy by difficulty (baseline vs latest) ----
function drawByDiff() {
  const box = document.getElementById("bydiff");
  const groups = Object.keys(L.by_difficulty);
  const series = P ? [["baseline", P, "--prev"], ["latest", L, "--cur"]] : [["latest", L, "--cur"]];
  document.getElementById("diff-sub").textContent = P ? `Baseline: ${short(P.label)} · Latest: ${short(L.label)}` : "Latest run";
  const f = frame(box, 240, {l: 40, r: 8, t: 12, b: 28});
  const y = yAxis(f, 0, 1, [0, .25, .5, .75, 1], t => t * 100 + "%");
  const gw = (f.x1 - f.x0) / groups.length, bw = Math.min(46, (gw - 28) / series.length);
  groups.forEach((g, gi) => {
    const cx = f.x0 + gw * gi + gw / 2;
    const t = svgEl("text", {x: cx, y: f.y0 + 16, "text-anchor": "middle"}); t.textContent = g; f.svg.appendChild(t);
    series.forEach(([name, run, tok], si) => {
      const v = run.by_difficulty[g]; if (!v) return;
      const x = cx - (series.length * bw + (series.length - 1) * 2) / 2 + si * (bw + 2), top = y(v.accuracy), h = f.y0 - top;
      const r = Math.min(4, h / 2);
      const p = svgEl("path", {d: `M${x},${f.y0} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${f.y0} Z`, fill: css(tok)});
      p.addEventListener("pointermove", ev => showTip(ev, `${g} · ${name}`, [["resolved", pct(v.accuracy)], ["instances", `${v.solved} / ${v.n}`], ["run", short(run.label)]]));
      p.addEventListener("pointerleave", hideTip); f.svg.appendChild(p);
      const lab = svgEl("text", {x: x + bw / 2, y: top - 4, "text-anchor": "middle", class: "lbl"}); lab.textContent = Math.round(v.accuracy * 100); f.svg.appendChild(lab);
    });
  });
  const legend = document.getElementById("diff-legend"); legend.replaceChildren();
  for (const [name, , tok] of series) { const item = el("span"); const sw = el("i"); sw.style.background = css(tok); item.append(sw, name); legend.appendChild(item); }
}

// ---- 3. failure modes (latest; baseline count in the label) ----
function drawModes() {
  const box = document.getElementById("modes");
  const modes = Object.entries(L.failure_modes);
  const total = modes.reduce((s, [, v]) => s + v, 0);
  document.getElementById("modes-sub").textContent = `${total} failed or errored sample(s) in the latest run` + (P ? "; baseline counts in parentheses" : "");
  const rowH = 26, f = frame(box, modes.length * rowH + 8, {l: 150, r: 70, t: 4, b: 4});
  const max = Math.max(1, ...modes.map(([, v]) => v), ...(P ? Object.values(P.failure_modes) : [0]));
  modes.forEach(([m, v], i) => {
    const yy = f.y1 + i * rowH, w = (f.x1 - f.x0) * v / max;
    const lab = svgEl("text", {x: f.x0 - 8, y: yy + rowH / 2 + 4, "text-anchor": "end", class: "lbl"}); lab.textContent = m; f.svg.appendChild(lab);
    f.svg.appendChild(svgEl("line", {x1: f.x0, x2: f.x0, y1: yy + 3, y2: yy + rowH - 3, stroke: css("--axis")}));
    if (v > 0) {
      const r = Math.min(4, w / 2), bh = 14, top = yy + (rowH - bh) / 2;
      const p = svgEl("path", {d: `M${f.x0},${top} H${f.x0 + w - r} Q${f.x0 + w},${top} ${f.x0 + w},${top + r} V${top + bh - r} Q${f.x0 + w},${top + bh} ${f.x0 + w - r},${top + bh} H${f.x0} Z`, fill: css("--cur")});
      p.addEventListener("pointermove", ev => showTip(ev, m, [["latest", String(v)], ...(P ? [["baseline", String(P.failure_modes[m])]] : [])]));
      p.addEventListener("pointerleave", hideTip); f.svg.appendChild(p);
    }
    const val = svgEl("text", {x: f.x0 + w + 6, y: yy + rowH / 2 + 4, class: "lbl"}); val.textContent = v + (P ? ` (${P.failure_modes[m]})` : ""); f.svg.appendChild(val);
  });
}

// ---- 4. flips ----
function drawFlips() {
  const box = document.getElementById("flips"); box.replaceChildren();
  const sub = document.getElementById("flips-sub");
  if (!D.flips) { sub.textContent = "No comparable earlier run to compare against."; return; }
  const F = D.flips;
  sub.textContent = `${F.common} instances in both runs: ${F.both} solved by both, ${F.neither} by neither. Mean ${pct(F.prev_mean)} → ${pct(F.latest_mean)}.`;
  for (const [title, ids, cls, sign] of [["Newly solved", F.fixed, "up", "▲"], ["Newly failed", F.broken, "down", "▼"]]) {
    const d = el("div"); d.appendChild(el("h2", {class: cls}, `${sign} ${title} (${ids.length})`));
    const ul = el("ul"); if (!ids.length) ul.appendChild(el("li", {class: "muted"}, "none"));
    for (const i of ids) ul.appendChild(el("li", {}, i)); d.appendChild(ul); box.appendChild(d);
  }
}

// ---- 5. instance x run matrix ----
let onlyChanged = false;
function drawMatrix() {
  const box = document.getElementById("matrix");
  const M = D.matrix, runs = M.runs.map(id => byId[id]);
  const inst = onlyChanged ? M.instances.filter(i => i.changed) : M.instances;
  const cw = 26, ch = 12, gap = 2, left = 300, top = 40;
  const W = left + runs.length * (cw + gap) + 20, H = top + inst.length * (ch + gap) + 8;
  box.replaceChildren();
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img"}); box.appendChild(svg);
  const shade = p => p == null ? css("--seq-0") : p >= 1 ? css("--seq-4") : p >= .67 ? css("--seq-3") : p >= .34 ? css("--seq-2") : p > 0 ? css("--seq-1") : css("--seq-0");
  runs.forEach((r, j) => {
    const x = left + j * (cw + gap) + cw / 2;
    const t = svgEl("text", {x, y: top - 22, "text-anchor": "middle", class: r.run_id === D.latest ? "lbl" : ""});
    t.textContent = "#" + (D.runs.indexOf(r) + 1); svg.appendChild(t);
    const k = svgEl("text", {x, y: top - 8, "text-anchor": "middle"}); k.textContent = r.kind === "teacher" ? "T" : "S"; svg.appendChild(k);
  });
  let lastDiff = null;
  inst.forEach((ins, i) => {
    const y = top + i * (ch + gap);
    const lab = svgEl("text", {x: left - 8, y: y + ch - 2, "text-anchor": "end", class: ins.changed ? "lbl" : ""});
    lab.textContent = (ins.difficulty !== lastDiff ? ins.difficulty + " · " : "") + ins.id; lastDiff = ins.difficulty; svg.appendChild(lab);
    runs.forEach((r, j) => {
      const c = (M.cells[r.run_id] || {})[ins.id];
      const x = left + j * (cw + gap);
      const rect = svgEl("rect", {x, y, width: cw, height: ch, rx: 2, fill: c ? shade(c.p) : "none",
        stroke: c ? (c.err ? css("--bad") : "none") : css("--grid"), "stroke-dasharray": c ? "" : "2 2"});
      rect.addEventListener("pointermove", ev => showTip(ev, ins.id, c ? [
        ["result", c.p == null ? "errored" : c.n > 1 ? `solved ${Math.round(c.p * c.n)}/${c.n}` : c.p ? "solved" : "failed"],
        ...(c.mode ? [["why", c.mode]] : []), ...(c.steps != null ? [["steps", String(c.steps)]] : []),
        ["run", short(r.label)], ["difficulty", ins.difficulty]] : [["result", "not in this run"], ["run", short(r.label)]]));
      rect.addEventListener("pointerleave", hideTip); svg.appendChild(rect);
    });
  });
  const lg = document.getElementById("matrix-legend"); lg.replaceChildren();
  for (const [name, tok] of [["failed", "--seq-0"], ["solved sometimes", "--seq-2"], ["solved", "--seq-4"]]) {
    const item = el("span"); const sw = el("i"); sw.style.background = css(tok); sw.style.border = "1px solid " + css("--ring"); item.append(sw, name); lg.appendChild(item); }
  const e = el("span"); const sw = el("i"); sw.style.border = "1.5px solid " + css("--bad"); e.append(sw, "errored"); lg.appendChild(e);
  lg.appendChild(el("span", {class: "muted"}, "S = student, T = teacher; # = run number in the table below"));
}
document.getElementById("only-changed").addEventListener("click", ev => {
  onlyChanged = !onlyChanged; ev.target.setAttribute("aria-pressed", onlyChanged);
  ev.target.textContent = onlyChanged ? "Show all" : "Show changed only"; drawMatrix(); });

// ---- 6. runs table ----
function drawRuns() {
  const t = document.getElementById("runs"); t.replaceChildren();
  const head = el("tr");
  for (const [h, num] of [["#", 1], ["Run", 0], ["Label", 0], ["Kind", 0], ["Resolved", 1], ["easy", 1], ["medium", 1], ["hard", 1], ["Steps", 1], ["Calls", 1], ["Cost", 1], ["Commit", 0], ["When", 0]])
    head.appendChild(el("th", num ? {class: "num"} : {}, h));
  t.appendChild(head);
  D.runs.slice().reverse().forEach(r => {
    const tr = el("tr", r.run_id === D.latest || r.run_id === D.previous ? {class: "hl"} : {});
    const bd = d => r.by_difficulty[d] ? pct(r.by_difficulty[d].accuracy, 0) : "–";
    const cells = [[D.runs.indexOf(r) + 1, 1], [r.run_id, 0], [r.label + (r.run_id === D.latest ? "  (latest)" : r.run_id === D.previous ? "  (baseline)" : ""), 0],
      [r.kind, 0], [`${pct(r.accuracy)} (${r.solved}/${r.n})`, 1], [bd("easy"), 1], [bd("medium"), 1], [bd("hard"), 1],
      [r.mean_steps == null ? "–" : r.mean_steps.toFixed(1), 1], [r.calls == null ? "–" : r.calls.toLocaleString(), 1],
      [r.cost == null ? "–" : "$" + r.cost.toFixed(2), 1], [(r.commit || "–") + (r.dirty ? "*" : ""), 0], [fmtTime(r.created_at), 0]];
    for (const [v, num] of cells) tr.appendChild(el("td", num ? {class: "num"} : {}, String(v)));
    t.appendChild(tr);
  });
}

function drawAll() { drawTrend(); drawByDiff(); drawModes(); drawFlips(); drawMatrix(); drawRuns(); }
drawAll();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawAll, 120); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
})();
</script>
</body>
</html>
"""
