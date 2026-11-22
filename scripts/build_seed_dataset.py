"""Build the seed dataset from scripts/seed_specs/batch_*.py.

For each instance spec we:
  1. materialize a small buggy repo under data/repos/<id>/ (source + existing passing tests),
  2. replay the engineer's action script through the real RepoEnv tools, recording the
     true observations (so trajectories are consistent with what an agent would see),
  3. take the resulting `git diff` as the gold patch,
  4. store the hidden tests as `test_patch`, split into FAIL_TO_PASS / PASS_TO_PASS by running
     them on the buggy repo.

This mirrors how the 10k engineer demos should be converted: keep their actions, re-execute
them in the sandbox to regenerate observations, and drop any demo whose patch doesn't validate.

Usage:
  python scripts/build_seed_dataset.py                                   # all batches -> data/demos/seed.jsonl
  python scripts/build_seed_dataset.py --batch batch_03 --out /tmp/b3.jsonl   # one batch
  repo-fix validate data/demos/seed.jsonl
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from repo_fix_agent.agent import run_agent  # noqa: E402
from repo_fix_agent.env import RepoEnv  # noqa: E402
from repo_fix_agent.grader import _run_tests  # noqa: E402
from repo_fix_agent.policies import ScriptedPolicy  # noqa: E402
from repo_fix_agent.schema import Demo, save_demos  # noqa: E402
from seed_specs import load_specs  # noqa: E402
from seed_specs._helpers import TEST_CMD, d  # noqa: E402

# --------------------------------------------------------------------------------------------
def write_repo(repo_dir: Path, files: dict[str, str]) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    for rel, content in files.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d(content) if content else "")


def make_test_patch(repo_dir: Path, rel: str, content: str) -> str:
    """Diff that adds the hidden test file, relative to the repo root."""
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / rel
        new.parent.mkdir(parents=True)
        new.write_text(content)
        r = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", rel],
                           cwd=tmp, capture_output=True, text=True)
        assert f"+++ b/{rel}" in r.stdout, r.stdout[:200]
        return r.stdout


def hidden_test_ids(rel: str, content: str) -> list[str]:
    return [f"{rel}::{line.split('(')[0][4:]}" for line in content.splitlines() if line.startswith("def test_")]


def build(specs: list[dict]) -> list[Demo]:
    ids = [s["id"] for s in specs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"duplicate instance ids: {sorted(dupes)}")
    demos = []
    for spec in specs:
        repo_rel = f"data/repos/{spec['id']}"
        repo_dir = ROOT / repo_rel
        write_repo(repo_dir, spec["files"])
        hidden_rel = f"tests/test_issue_{spec['id'].split('-')[1]}.py"
        hidden = d(spec["hidden_test"])
        issue = d(spec["issue"])

        with RepoEnv(repo_dir) as env:
            res = run_agent(ScriptedPolicy(spec["steps"]), env, issue, TEST_CMD)
        assert res.submitted, spec["id"]
        for s in res.trajectory:
            if s.observation.startswith("Error:") or "command not found" in s.observation:
                raise RuntimeError(f"{spec['id']}: step failed: {s.action} -> {s.observation}")

        # Split hidden tests the SWE-bench way: tests that already pass on the buggy repo are
        # regression guards (PASS_TO_PASS), the rest must flip (FAIL_TO_PASS).
        test_patch = make_test_patch(repo_dir, hidden_rel, hidden)
        with RepoEnv(repo_dir) as env:
            env.apply_patch(test_patch)
            before = _run_tests(env, TEST_CMD, hidden_test_ids(hidden_rel, hidden))
            existing = _run_tests(env, TEST_CMD, spec["p2p"])
        broken = [t for t, ok in existing.items() if not ok]
        if broken:
            raise RuntimeError(f"{spec['id']}: existing tests listed in p2p fail on the buggy repo: {broken}")
        f2p = [t for t, ok in before.items() if not ok]
        p2p = spec["p2p"] + [t for t, ok in before.items() if ok]
        assert f2p, f"{spec['id']}: no hidden test fails on the buggy repo"

        files_changed = sorted({l[6:] for l in res.patch.splitlines() if l.startswith("+++ b/")})
        demos.append(Demo(
            instance_id=spec["id"],
            repo=repo_rel,
            base_commit=None,
            problem_statement=issue,
            test_cmd=TEST_CMD,
            test_patch=test_patch,
            FAIL_TO_PASS=f2p,
            PASS_TO_PASS=p2p,
            patch=res.patch,
            trajectory=res.trajectory,
            metadata={
                "source": "engineer_seed",
                "bug_type": spec["bug_type"],
                "difficulty": spec["difficulty"],
                "files_changed": files_changed,
                "num_steps": len(res.trajectory),
                "language": "python",
            },
        ))
        print(f"built {spec['id']}: {len(res.trajectory)} steps, files={files_changed}")
    return demos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", help="batch module name(s), e.g. batch_03; default: all")
    ap.add_argument("--out", default=str(ROOT / "data/demos/seed.jsonl"))
    a = ap.parse_args()
    out = Path(a.out)
    save_demos(build(load_specs(a.batch)), out)
    print(f"wrote {out}")
