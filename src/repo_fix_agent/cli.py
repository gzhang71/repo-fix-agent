"""repo-fix CLI.

  repo-fix fix      --repo PATH --issue-file ISSUE.md [--model M] [--apply]   # plug onto any repo
  repo-fix validate data/demos/seed.jsonl                                     # dataset QA
  repo-fix eval     data/demos/seed.jsonl --model M                           # success rate
  repo-fix export-sft data/demos/seed.jsonl out.jsonl                         # training data
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .agent import run_agent
from .env import RepoEnv
from .grader import grade, validate_instance
from .policies import make_policy
from .schema import load_demos
from .sft import export_sft


def _policy(model: str):
    try:
        return make_policy(model)
    except ValueError as e:
        raise SystemExit(str(e))


def cmd_fix(a) -> int:
    issue = Path(a.issue_file).read_text() if a.issue_file else a.issue
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
    root = Path(a.root or ".")
    demos = load_demos(a.demos)
    resolved = 0
    for d in demos:
        repo = d.repo if "://" in d.repo else str(root / d.repo)
        with RepoEnv(repo, d.base_commit) as env:
            res = run_agent(_policy(a.model), env, d.problem_statement, d.test_cmd, a.max_steps)
        g = grade(d, res.patch, root)
        resolved += g.resolved
        print(json.dumps({"instance_id": d.instance_id, "resolved": g.resolved, "steps": len(res.trajectory)}))
    print(f"resolved {resolved}/{len(demos)} = {resolved / max(1, len(demos)):.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="repo-fix")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fix")
    f.add_argument("--repo", required=True)
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--issue")
    g.add_argument("--issue-file")
    f.add_argument("--test-cmd", default=None)
    f.add_argument("--model", default="claude-opus-5-5")
    f.add_argument("--max-steps", type=int, default=40)
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
    e.add_argument("--max-steps", type=int, default=40)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("export-sft")
    s.add_argument("demos")
    s.add_argument("out")
    s.set_defaults(fn=lambda a: print(f"wrote {export_sft(a.demos, a.out)} examples") or 0)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
