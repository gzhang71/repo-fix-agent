"""repo-fix CLI.

  repo-fix fix      --repo PATH --issue-file ISSUE.md [--model M] [--apply]   # plug onto any repo
  repo-fix validate data/demos/seed.jsonl                                     # dataset QA
  repo-fix eval     data/demos/seed.jsonl --model M                           # success rate
  repo-fix export-sft data/demos/seed.jsonl out.jsonl                         # training data

Data pipeline (docs/DEMO_DATA_PROPOSALS.md): audit, split, eval-set, distill, mix, sft-mix,
aux-tasks, prefs, verifier-train, best-of-n, rl-rollouts, retrieval-index, retrieve, augment.
Run `repo-fix <command> -h` for options.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .agent import run_agent
from .env import RepoEnv
from .envfile import load_env
from .grader import grade, validate_instance
from .policies import make_policy
from .schema import load_demos
from .sft import export_sft


PIPELINE_MODULES = ["audit", "split", "distill", "sft_data", "aux_tasks", "preferences",
                    "verifier", "rl", "retrieval", "augment", "onpolicy", "training",
                    "difficulty", "teacher_signal", "retention"]


def _policy(model: str):
    try:
        return make_policy(model)
    except ValueError as e:
        raise SystemExit(str(e))


def cmd_fix(a) -> int:
    issue = Path(a.issue_file).read_text() if a.issue_file else a.issue
    if a.max_steps is None:
        from .policies import StudentServingConfig
        a.max_steps = StudentServingConfig.load().max_steps if a.model.startswith("student") else 40
    with RepoEnv(a.repo) as env:
        res = run_agent(_policy(a.model), env, issue, a.test_cmd, a.max_steps)
    print(res.patch or "(no changes)")
    if a.apply and res.patch:
        subprocess.run(["git", "apply", "-"], cwd=a.repo, input=res.patch, text=True, check=True)
        print(f"Applied to {a.repo}", file=sys.stderr)
    return 0 if res.submitted else 1


def cmd_validate(a) -> int:
    root = Path(a.root or ".")
    bad = 0
    for d in load_demos(a.demos):
        problems = validate_instance(d, root)
        print(f"{'FAIL' if problems else 'ok  '}  {d.instance_id}  ({len(d.trajectory)} steps)")
        for p in problems:
            print(f"      {p}")
        bad += bool(problems)
    print(f"\n{bad} invalid instance(s)")
    return 1 if bad else 0


def cmd_eval(a) -> int:
    from .evaluation import evaluate, format_summary, summarize
    from .llm_cache import enabled, get_cache

    root = Path(a.root or ".")
    demos = load_demos(a.demos)
    if a.ids:
        want = set(a.ids.split(","))
        demos = [d for d in demos if d.instance_id in want]
    demos = demos[:a.limit] if a.limit else demos
    if (a.samples > 1 or a.best_of > 1) and a.model.startswith("student") and not a.temperature:
        from .policies import StudentServingConfig
        if StudentServingConfig.load().temperature == 0:
            raise SystemExit("--samples > 1 needs sampling: pass --temperature (e.g. 0.7) for the student")
    if a.max_steps is None:
        from .policies import StudentServingConfig
        a.max_steps = StudentServingConfig.load().max_steps if a.model.startswith("student") else 40
    factory = lambda: make_policy(a.model, a.temperature)  # noqa: E731
    try:
        factory()  # fail fast on an unknown model id
    except ValueError as e:
        raise SystemExit(str(e))
    out = Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not a.resume:
            out.write_text("")
    lock = __import__("threading").Lock()
    kind = "student" if a.model.startswith("student") else "teacher"
    model_id = a.model
    if a.model.startswith("student"):
        from .policies import StudentServingConfig
        model_id = a.model.split(":", 1)[1] if ":" in a.model else StudentServingConfig.load().model_name

    db = run_id = None
    skip: set[tuple[str, int]] = set()
    if not a.no_db:
        from .rundb import RunDB

        db = RunDB(a.db)
        db.upsert_instances(demos)
        if a.resume:
            run_id = db.resolve(a.resume)
            skip = db.done_keys(run_id)
            db._exec("UPDATE runs SET status='running' WHERE run_id=?", (run_id,))
            print(f"resuming run {run_id}: {len(skip)} sample(s) already done", file=sys.stderr)
        else:
            run_id = db.start_run(model=model_id, kind=kind, dataset=a.demos, label=a.label, n_instances=len(demos),
                                  samples=a.samples, best_of=a.best_of, temperature=a.temperature,
                                  max_steps=a.max_steps, sandbox=_sandbox_desc(a), scaffold=_scaffold(a, factory),
                                  argv=sys.argv[1:])
            print(f"run {run_id} -> {db.path}", file=sys.stderr)
    elif a.resume:
        raise SystemExit("--resume needs the run database (drop --no-db)")

    def on_result(rec):
        with lock:
            print(json.dumps(rec), flush=True)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        if db:
            db.add_result(run_id, rec)

    rollouts_f = Path(a.save_rollouts).open("a" if a.resume else "w") if a.save_rollouts else None

    def on_rollout(demo):
        if rollouts_f:
            with lock:
                rollouts_f.write(json.dumps(demo.to_dict()) + "\n")
                rollouts_f.flush()
        if db:
            db.add_episode(run_id, demo)

    try:
        records = evaluate(demos, factory, root, a.max_steps, a.workers, on_result, samples=a.samples,
                           source=f"{kind}:{a.model}", on_rollout=on_rollout if (rollouts_f or db) else None,
                           best_of=a.best_of, skip=skip)
    except BaseException:
        if db:
            db.finish_run(run_id, status="failed")
        raise
    finally:
        if rollouts_f:
            rollouts_f.close()
    usage = get_cache().usage.get(model_id) if enabled() else None
    if db:   # on resume, summarize every sample of the run, not just this session's
        records = db.results(run_id)
        prev = (db.run(run_id).get("usage") or {}) if a.resume else {}
        if prev and usage:
            usage = {k: (prev.get(k) or 0) + (usage.get(k) or 0) for k in set(prev) | set(usage)}
        usage = usage or prev or None
    summary = summarize(model_id, records, usage)
    if out:
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    if db:
        db.finish_run(run_id, summary, usage)
    print(format_summary(summary))
    if run_id:
        print(f"run {run_id} saved to the run database (repo-fix db show {run_id})")
    if db and not a.no_dashboard:   # every backtest refreshes the dashboard
        from .dashboard import write_dashboard
        try:
            path = write_dashboard(db, a.dashboard, run=run_id)
            print(f"dashboard: {path.resolve()}")
        except Exception as e:  # noqa: BLE001 - a dashboard problem must not fail the eval
            print(f"dashboard not written: {type(e).__name__}: {e}", file=sys.stderr)
    if db:
        db.close()
    return 0


def _sandbox_desc(a) -> str:
    import os
    return f"{a.sandbox or os.environ.get('REPOFIX_SANDBOX') or 'auto'}:" \
           f"{a.sandbox_image or os.environ.get('REPOFIX_SANDBOX_IMAGE') or 'repo-fix-sandbox:py3.12'}"


def _scaffold(a, factory) -> dict:
    """What shapes the agent's behaviour, so runs with different scaffolds can be told apart."""
    import dataclasses
    import hashlib

    pol = factory()
    opts = getattr(pol, "agent_options", None)
    from .agent import AgentOptions
    opts = opts if isinstance(opts, AgentOptions) else AgentOptions()
    out = {"policy": type(pol).__name__, "system_prompt_sha": hashlib.sha256(opts.system_prompt().encode()).hexdigest()[:12],
           "agent_options": dataclasses.asdict(opts)}
    cfg = getattr(pol, "cfg", None)
    if cfg is not None and dataclasses.is_dataclass(cfg):
        out["serving"] = {k: v for k, v in dataclasses.asdict(cfg).items() if "key" not in k}
    for k in ("effort", "max_tokens"):
        if hasattr(pol, k):
            out[k] = getattr(pol, k)
    return out


def main(argv: list[str] | None = None) -> int:
    load_env()   # .env -> os.environ (API keys); existing env vars win
    p = argparse.ArgumentParser(prog="repo-fix")
    p.add_argument("--sandbox", choices=["auto", "docker", "local"], default=None,
                   help="where agent commands and tests run (default: $REPOFIX_SANDBOX or auto = "
                        "docker if the daemon and sandbox image are available, else local)")
    p.add_argument("--sandbox-image", default=None,
                   help="docker image for the sandbox (default: $REPOFIX_SANDBOX_IMAGE or "
                        "repo-fix-sandbox:py3.12, built from docker/sandbox.Dockerfile)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fix")
    f.add_argument("--repo", required=True)
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--issue")
    g.add_argument("--issue-file")
    f.add_argument("--test-cmd", default=None)
    f.add_argument("--model", default="claude-opus-5-5")
    f.add_argument("--max-steps", type=int, default=None,
                   help="step budget (default: student.serving.max_steps for the student, 40 for teachers)")
    f.add_argument("--apply", action="store_true", help="apply the resulting diff to --repo")
    f.set_defaults(fn=cmd_fix)

    v = sub.add_parser("validate")
    v.add_argument("demos")
    v.add_argument("--root", default=None, help="dir that demo.repo paths are relative to (default: cwd)")
    v.set_defaults(fn=cmd_validate)

    e = sub.add_parser("eval")
    e.add_argument("demos")
    e.add_argument("--model", default="claude-opus-5-5")
    e.add_argument("--root", default=None)
    e.add_argument("--max-steps", type=int, default=None,
                   help="step budget (default: student.serving.max_steps for the student, 40 for teachers)")
    e.add_argument("--workers", type=int, default=1, help="instances evaluated concurrently")
    e.add_argument("--out", default=None, help="per-instance JSONL (+ .summary.json next to it)")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--ids", default=None, help="comma-separated instance ids")
    e.add_argument("--samples", type=int, default=1, help="episodes per instance (mean pass@1 and pass@k)")
    e.add_argument("--label", default=None, help="name for this run in the run database")
    e.add_argument("--db", default=None, help="run database path (default $REPOFIX_DB or runs/repofix.sqlite)")
    e.add_argument("--no-db", action="store_true", help="don't record this run in the run database")
    e.add_argument("--resume", default=None, help="continue a run (id/prefix/label): skip samples already done")
    e.add_argument("--dashboard", default="runs/dashboard.html", help="backtest dashboard written after the run")
    e.add_argument("--no-dashboard", action="store_true", help="don't regenerate the dashboard")
    e.add_argument("--best-of", type=int, default=1,
                   help="sample N candidates per instance, pick one with the execution reranker, grade it "
                        "(also reports the oracle: any candidate correct)")
    e.add_argument("--temperature", type=float, default=None, help="student sampling temperature override")
    e.add_argument("--save-rollouts", default=None,
                   help="write every episode as a Demo JSONL (input for prefs / onpolicy-distill)")
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("export-sft")
    s.add_argument("demos")
    s.add_argument("out")
    s.set_defaults(fn=lambda a: print(f"wrote {export_sft(a.demos, a.out)} examples") or 0)

    for name in PIPELINE_MODULES:  # docs/DEMO_DATA_PROPOSALS.md, one module per proposal
        importlib.import_module(f".pipeline.{name}", __package__).register(sub)
    importlib.import_module(".web.server", __package__).register(sub)   # repo-fix serve
    importlib.import_module(".llm_cache", __package__).register(sub)    # repo-fix llm-cache
    importlib.import_module(".rundb", __package__).register(sub)        # repo-fix db
    importlib.import_module(".dashboard", __package__).register(sub)    # repo-fix dashboard

    a = p.parse_args(argv)
    if a.sandbox:  # read by make_sandbox(), so every command and pipeline step honours it
        os.environ["REPOFIX_SANDBOX"] = a.sandbox
    if a.sandbox_image:
        os.environ["REPOFIX_SANDBOX_IMAGE"] = a.sandbox_image
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
