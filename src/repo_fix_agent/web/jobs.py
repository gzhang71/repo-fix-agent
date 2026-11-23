"""Repo-fix jobs: clone -> environment -> find bugs -> fix each in a sandbox -> verify -> PRs.

Bug sources
  * failing tests: run the repo's suite in the sandbox; failing tests are grouped by test file and
    each group becomes one bug (one agent episode, one PR). These are *verifiable*: a fix counts
    only if the group's tests pass afterwards and no test that passed before now fails.
  * GitHub issues labeled `bug` (optional): the issue text is the problem statement. There is no
    failing test, so such fixes are only checked for regressions and are marked "unverified".

Each bug is fixed independently from the base commit, so every PR applies on its own. PRs are
opened only for the user's request (or automatically for *verified* fixes when auto_pr is set).
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent import run_agent
from ..env import RepoEnv
from ..pipeline.parallel import pmap
from .forge import ForgeError, LocalForge, RepoRef, git, make_forge, parse_repo_url

DEFAULT_TEST_CMD = "python -m pytest -q -p no:cacheprovider"
_OUTCOME = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED) (\S+)")
_SECTION = re.compile(r"^_{3,} (.+?) _{3,}$")


# ---- test-output parsing ----------------------------------------------------------------------
@dataclass
class TestRun:
    returncode: int
    passed: set[str]
    failed: set[str]           # FAILED + ERROR (collection errors use the file as id)
    output: str
    timed_out: bool = False

    @property
    def parsed(self) -> bool:
        return bool(self.passed or self.failed)


def pytest_cmd(test_cmd: str, ids: list[str] | None = None) -> str:
    cmd = test_cmd
    if "pytest" in cmd:
        cmd += " -rA --tb=short --color=no"
    if ids:
        cmd += " " + " ".join(_q(i) for i in ids)
    return cmd


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def parse_pytest(output: str, returncode: int = 0, timed_out: bool = False) -> TestRun:
    passed, failed = set(), set()
    for line in output.splitlines():
        m = _OUTCOME.match(line)
        if m:
            (failed if m.group(1) in ("FAILED", "ERROR") else passed if m.group(1) != "SKIPPED" else set()).add(
                m.group(2))
    return TestRun(returncode, passed, failed, output, timed_out)


_LOG = ".git/repofix-tests.log"


def run_suite(env: RepoEnv, cmd: str, timeout: int) -> TestRun:
    """Run tests with output going to a file: exec output is capped (8 KB) and big suites would
    lose the failure details we put in the problem statement."""
    r = env.run(f"( {cmd} ) > {_LOG} 2>&1", timeout=timeout)
    try:
        out = env.sandbox.read_file(_LOG)
    except (FileNotFoundError, ValueError):
        out = r.output
    return parse_pytest(out, r.returncode, r.timed_out)


def failure_sections(output: str) -> dict[str, str]:
    """Map pytest section headers (e.g. 'test_add' or 'TestX.test_y') to their traceback text."""
    out, cur, buf = {}, None, []
    for line in output.splitlines():
        m = _SECTION.match(line)
        if m or line.startswith("=" * 10):
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = (m.group(1).strip(), []) if m else (None, [])
            continue
        if cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def _section_for(test_id: str, sections: dict[str, str]) -> str:
    tail = test_id.split("::")[1:]
    for key in (".".join(tail), tail[-1] if tail else "", test_id, f"ERROR collecting {test_id}"):
        if key and key in sections:
            return sections[key]
    return ""


# ---- model ------------------------------------------------------------------------------------
@dataclass
class JobConfig:
    repo_url: str
    branch: str | None = None
    model: str = "claude-opus-5-5"
    test_cmd: str = DEFAULT_TEST_CMD
    setup_cmd: str | None = None
    sources: list[str] = field(default_factory=lambda: ["tests"])   # "tests", "issues"
    max_bugs: int = 10
    max_steps: int = 40
    workers: int = 2
    auto_pr: bool = False
    test_timeout: int = 900

    @classmethod
    def from_dict(cls, d: dict) -> "JobConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in known and v not in (None, "")})
        cfg.max_bugs = max(1, min(int(cfg.max_bugs), 50))
        cfg.max_steps = max(5, min(int(cfg.max_steps), 150))
        cfg.workers = max(1, min(int(cfg.workers), 8))
        cfg.sources = [s for s in cfg.sources if s in ("tests", "issues")] or ["tests"]
        return cfg


@dataclass
class Bug:
    id: str
    title: str
    source: str                      # tests | issue
    problem_statement: str
    tests: list[str] = field(default_factory=list)
    issue_number: int | None = None
    status: str = "queued"           # queued fixing verifying verified unverified regressed no_fix failed cancelled
    steps: int = 0
    last_action: str = ""
    patch: str = ""
    summary: str = ""
    verification: dict = field(default_factory=dict)
    error: str | None = None
    pr: dict | None = None           # {"url", "number", "branch", "state": opening|open|failed, "error"}


@dataclass
class Job:
    id: str
    config: JobConfig
    status: str = "queued"           # queued cloning setup discovering fixing done failed cancelled
    created: float = field(default_factory=time.time)
    finished: float | None = None
    repo: dict = field(default_factory=dict)       # slug, is_github, base_branch, base_sha, can_open_prs
    baseline: dict = field(default_factory=dict)   # passed/failed counts, failing ids
    bugs: list[Bug] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    image: str | None = None

    def to_dict(self, since: int = 0, full: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d["events"] = [e for e in d["events"] if e["seq"] > since][-500:]
        if not full:
            d.pop("bugs")
            d["n_bugs"] = len(self.bugs)
            d["n_verified"] = sum(b.status == "verified" for b in self.bugs)
            d["n_prs"] = sum(bool(b.pr and b.pr.get("url")) for b in self.bugs)
            d.pop("events")
        return d


class Cancelled(Exception):
    pass


# ---- store ------------------------------------------------------------------------------------
class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        for p in sorted(self.root.glob("*/job.json")):
            try:
                d = json.loads(p.read_text())
                job = Job(id=d["id"], config=JobConfig.from_dict(d["config"]),
                          **{k: d[k] for k in ("status", "created", "finished", "repo", "baseline", "events",
                                               "error", "image") if k in d})
                job.bugs = [Bug(**b) for b in d.get("bugs", [])]
                if job.status not in ("done", "failed", "cancelled"):
                    job.status, job.error = "failed", "server restarted while the job was running"
                for b in job.bugs:
                    if b.status in ("queued", "fixing", "verifying"):
                        b.status = "cancelled"
                    if b.pr and b.pr.get("state") == "opening":
                        b.pr = {"state": "failed", "error": "server restarted"}
                self.jobs[job.id] = job
            except Exception:  # noqa: BLE001 - a corrupt job file must not stop the server
                continue

    def dir(self, job: Job) -> Path:
        return self.root / job.id

    def save(self, job: Job) -> None:
        with self.lock:
            d = self.dir(job)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "job.json.tmp"
            tmp.write_text(json.dumps(job.to_dict(), default=list))
            tmp.replace(d / "job.json")

    def create(self, cfg: JobConfig) -> Job:
        job = Job(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], config=cfg)
        with self.lock:
            self.jobs[job.id] = job
        self.save(job)
        return job

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: -j.created)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


# ---- runner -----------------------------------------------------------------------------------
PolicyFactory = Callable[[str], Any]
ForgeFactory = Callable[[RepoRef, Path], LocalForge]


def check_model_credentials(model: str) -> None:
    """SDK clients only notice a missing key on the first request; fail before cloning instead."""
    import importlib.util
    import os

    if model.startswith("claude-"):
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment of `repo-fix serve`")
    elif model.startswith("gpt-") or (model[:1] == "o" and model[1:2].isdigit()):
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("the openai package is not installed (pip install -e '.[openai]')")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment of `repo-fix serve`")
    elif model == "student" or model.startswith("student:"):
        from ..policies import StudentServingConfig

        cfg = StudentServingConfig.load()
        local = any(h in cfg.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not local and not os.environ.get(cfg.api_key_env):
            raise RuntimeError(f"{cfg.api_key_env} is not set (student is served from {cfg.base_url}); "
                               f"add it to .env")


def _default_policy_factory(model: str):
    from ..policies import make_policy
    return make_policy(model)


class JobRunner:
    def __init__(self, store: JobStore, backend: str = "docker", *, policy_factory: PolicyFactory | None = None,
                 forge_factory: ForgeFactory | None = None, build_image: Callable[..., str] | None = None,
                 max_jobs: int = 2):
        self.store, self.backend = store, backend
        self.policy_factory = policy_factory or _default_policy_factory
        self.check_credentials = check_model_credentials if policy_factory is None else (lambda model: None)
        self.forge_factory = forge_factory or make_forge
        if build_image is None and backend == "docker":
            from .envsetup import build_env_image as build_image
        self.build_image = build_image
        self.pool = ThreadPoolExecutor(max_jobs, thread_name_prefix="job")
        self.pr_pool = ThreadPoolExecutor(2, thread_name_prefix="pr")   # PR clicks don't wait behind jobs
        self._cancel: dict[str, threading.Event] = {}
        self._repo_locks: dict[str, threading.Lock] = {}
        self._forges: dict[str, LocalForge] = {}

    # -- public -------------------------------------------------------------------------------
    def submit(self, cfg: JobConfig) -> Job:
        ref = parse_repo_url(cfg.repo_url)   # raises ValueError for junk before a job exists
        job = self.store.create(cfg)
        job.repo = {"slug": ref.slug, "url": ref.url, "is_github": ref.is_github}
        self._cancel[job.id] = threading.Event()
        self._repo_locks[job.id] = threading.Lock()
        self._log(job, f"queued {ref.slug} with {cfg.model}")
        self.pool.submit(self._run, job)
        return job

    def cancel(self, job: Job) -> None:
        ev = self._cancel.get(job.id)
        if ev:
            ev.set()
            self._log(job, "cancel requested")

    def open_pr(self, job: Job, bug: Bug, wait: bool = False) -> None:
        if not bug.patch:
            raise ValueError("this bug has no patch")
        if bug.pr and bug.pr.get("state") in ("opening", "open"):
            return
        bug.pr = {"state": "opening"}
        self.store.save(job)
        if wait:
            self._open_pr(job, bug)
        else:
            self.pr_pool.submit(self._open_pr, job, bug)

    # -- internals ----------------------------------------------------------------------------
    def _log(self, job: Job, msg: str, bug: Bug | None = None) -> None:
        with self.store.lock:
            seq = job.events[-1]["seq"] + 1 if job.events else 1
            job.events.append({"seq": seq, "t": time.time(), "msg": msg, "bug": bug.id if bug else None})
            if len(job.events) > 3000:
                del job.events[:1000]

    def _stage(self, job: Job, status: str, msg: str) -> None:
        self._check(job)
        job.status = status
        self._log(job, msg)
        self.store.save(job)

    def _check(self, job: Job) -> None:
        ev = self._cancel.get(job.id)
        if ev and ev.is_set():
            raise Cancelled()

    def _env(self, job: Job) -> RepoEnv:
        return RepoEnv(self.store.dir(job) / "repo", backend=self.backend, image=job.image)

    def _run_tests(self, job: Job, ids: list[str] | None = None, patch: str = "") -> TestRun:
        with self._env(job) as env:
            if patch:
                env.apply_patch(patch)
            return run_suite(env, pytest_cmd(job.config.test_cmd, ids), job.config.test_timeout)

    def _run(self, job: Job) -> None:
        cfg = job.config
        try:
            try:
                self.check_credentials(cfg.model)
                self.policy_factory(cfg.model)   # fail fast on a missing API key / bad model id
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"cannot use model {cfg.model!r}: {e}") from None
            ref = parse_repo_url(cfg.repo_url)
            workdir = self.store.dir(job)
            workdir.mkdir(parents=True, exist_ok=True)
            forge = self.forge_factory(ref, workdir)
            self._forges[job.id] = forge

            self._stage(job, "cloning", f"cloning {ref.slug}")
            repo = workdir / "repo"
            if repo.exists():
                shutil.rmtree(repo)
            base_branch = forge.clone(repo, cfg.branch)
            job.repo.update(base_branch=base_branch, base_sha=git(repo, "rev-parse", "HEAD").strip(),
                            can_open_prs=forge.can_open_prs)
            self._log(job, f"cloned {base_branch} @ {job.repo['base_sha'][:10]}")

            if self.backend == "docker" and self.build_image is not None:
                self._stage(job, "setup", "installing dependencies into a sandbox image (network on, one time)")
                job.image = self.build_image(repo, f"repo-fix-env:{job.id}", cfg.setup_cmd)
                self._log(job, f"environment image {job.image} ready")

            self._stage(job, "discovering", "running the test suite to find failing tests")
            bugs = self._discover(job, forge)
            job.bugs = bugs[:cfg.max_bugs]
            if len(bugs) > cfg.max_bugs:
                self._log(job, f"found {len(bugs)} bugs; fixing the first {cfg.max_bugs} (raise 'max bugs' for more)")
            if not job.bugs:
                job.status, job.finished = "done", time.time()
                self._log(job, "no bugs found: the test suite passes" +
                          (" and there are no open bug issues" if "issues" in cfg.sources else ""))
                self.store.save(job)
                return

            self._stage(job, "fixing", f"fixing {len(job.bugs)} bug(s), {cfg.workers} at a time")
            pmap(lambda b: self._fix(job, b), job.bugs, cfg.workers)
            self._check(job)

            if cfg.auto_pr and forge.can_open_prs:
                for b in job.bugs:
                    if b.status == "verified":
                        self.open_pr(job, b, wait=True)
            job.status, job.finished = "done", time.time()
            n = sum(b.status == "verified" for b in job.bugs)
            self._log(job, f"done: {n}/{len(job.bugs)} fixes verified")
        except Cancelled:
            job.status, job.finished = "cancelled", time.time()
            for b in job.bugs:
                if b.status in ("queued", "fixing", "verifying"):
                    b.status = "cancelled"
            self._log(job, "cancelled")
        except Exception as e:  # noqa: BLE001 - report every failure in the UI
            job.status, job.finished, job.error = "failed", time.time(), f"{type(e).__name__}: {e}"
            self._log(job, f"failed: {job.error}")
            (self.store.dir(job) / "error.txt").write_text(traceback.format_exc())
        finally:
            self.store.save(job)

    # -- discovery ----------------------------------------------------------------------------
    def _discover(self, job: Job, forge: LocalForge) -> list[Bug]:
        cfg, bugs = job.config, []
        if "tests" in cfg.sources:
            run = self._run_tests(job)
            job.baseline = {"passed": len(run.passed), "failed": sorted(run.failed), "returncode": run.returncode,
                            "timed_out": run.timed_out}
            self._log(job, f"baseline: {len(run.passed)} passed, {len(run.failed)} failing")
            if run.timed_out:
                raise RuntimeError(f"test suite timed out after {cfg.test_timeout}s")
            if not run.parsed and run.returncode not in (0, 5):
                raise RuntimeError("could not read test results; is the test command right? Last output:\n"
                                   + run.output[-1500:])
            bugs += self._bugs_from_failures(run)
        if "issues" in cfg.sources:
            if not hasattr(forge, "bug_issues"):
                self._log(job, "skipping issues: only supported for GitHub repos")
            else:
                for issue in forge.bug_issues(limit=cfg.max_bugs):
                    n = issue["number"]
                    bugs.append(Bug(id=f"i{n}", title=issue["title"][:120], source="issue", issue_number=n,
                                    problem_statement=f"{issue['title']}\n\n{issue.get('body') or ''}".strip()))
                self._log(job, f"{sum(b.source == 'issue' for b in bugs)} open issue(s) labeled bug")
        return bugs

    def _bugs_from_failures(self, run: TestRun) -> list[Bug]:
        sections = failure_sections(run.output)
        groups: dict[str, list[str]] = {}
        for tid in sorted(run.failed):
            groups.setdefault(tid.split("::")[0], []).append(tid)
        bugs = []
        for i, (path, ids) in enumerate(sorted(groups.items()), 1):
            detail = "\n\n".join(s for s in (_section_for(t, sections) for t in ids[:5]) if s)[:6000]
            names = ", ".join(t.split("::")[-1] for t in ids[:3]) + (f" (+{len(ids) - 3} more)" if len(ids) > 3 else "")
            ps = ("The following tests fail on the current code:\n\n" + "\n".join(f"  - {t}" for t in ids)
                  + (f"\n\nFailure output:\n```\n{detail}\n```" if detail else "")
                  + "\n\nFind and fix the underlying bug in the source code. Do not modify, skip or delete "
                    "the tests.")
            bugs.append(Bug(id=f"t{i}", title=f"Fix failing tests in {path}: {names}", source="tests",
                            problem_statement=ps, tests=ids))
        return bugs

    # -- fixing -------------------------------------------------------------------------------
    def _fix(self, job: Job, bug: Bug) -> None:
        cfg = job.config
        try:
            self._check(job)
            bug.status = "fixing"
            self._log(job, f"agent started on: {bug.title}", bug)
            self.store.save(job)

            def on_step(i, step):
                bug.steps = i + 1
                args = step.action.args or {}
                detail = args.get("command") or args.get("path") or args.get("pattern") or ""
                bug.last_action = f"{step.action.tool} {str(detail)[:120]}".strip()
                self._log(job, f"step {i + 1}: {bug.last_action}", bug)
                self._check(job)

            with self._env(job) as env:
                res = run_agent(self.policy_factory(cfg.model), env, bug.problem_statement, cfg.test_cmd,
                                cfg.max_steps, on_step=on_step)
            bug.patch = res.patch
            bug.summary = next((s.thought for s in reversed(res.trajectory) if s.thought), "")[:2000]
            if not res.patch.strip():
                bug.status = "no_fix"
                self._log(job, "agent finished without changing source code", bug)
                return
            bug.status = "verifying"
            self._log(job, f"verifying patch ({res.patch.count(chr(10))} diff lines)", bug)
            self.store.save(job)
            bug.verification = self._verify(job, bug)
            v = bug.verification
            if v["new_failures"]:
                bug.status = "regressed"
            elif bug.source == "tests":
                bug.status = "verified" if v["targets_pass"] else "failed"
            else:
                bug.status = "unverified"
            self._log(job, f"{bug.status}: targets {'pass' if v.get('targets_pass') else 'n/a or failing'}, "
                           f"{len(v['new_failures'])} new failure(s)", bug)
        except Cancelled:
            bug.status = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001
            bug.status, bug.error = "failed", f"{type(e).__name__}: {e}"
            self._log(job, f"error: {bug.error}", bug)
        finally:
            self.store.save(job)

    def _verify(self, job: Job, bug: Bug) -> dict:
        full = self._run_tests(job, patch=bug.patch)
        before_failed = set(job.baseline.get("failed", []))

        def fixed(t: str) -> bool:
            if t in full.failed:
                return False
            if "::" in t:
                return t in full.passed
            return not any(f.startswith(t) for f in full.failed)   # collection error on a whole file

        targets = {t: fixed(t) for t in bug.tests}
        return {"targets": targets, "targets_pass": all(targets.values()) if targets else None,
                "new_failures": sorted(full.failed - before_failed), "passed": len(full.passed),
                "failed": len(full.failed), "timed_out": full.timed_out}

    # -- PRs ----------------------------------------------------------------------------------
    def _open_pr(self, job: Job, bug: Bug) -> None:
        forge = self._forges.get(job.id) or self.forge_factory(parse_repo_url(job.config.repo_url), self.store.dir(job))
        repo = self.store.dir(job) / "repo"
        slug = re.sub(r"[^a-z0-9]+", "-", bug.title.lower()).strip("-")[:40]
        branch = f"repo-fix/{job.id[-6:]}-{bug.id}-{slug}"
        try:
            with self._repo_locks.setdefault(job.id, threading.Lock()):
                base = job.repo["base_branch"]
                git(repo, "checkout", "--quiet", "-B", branch, job.repo["base_sha"])
                try:
                    patch_file = repo / ".git" / "repofix.patch"
                    patch_file.write_text(bug.patch)
                    git(repo, "apply", "--whitespace=nowarn", str(patch_file))
                    git(repo, "add", "-A")
                    ident = [] if _has_identity(repo) else ["-c", "user.name=repo-fix-agent",
                                                            "-c", "user.email=repo-fix-agent@users.noreply.github.com"]
                    git(repo, *ident, "commit", "--quiet", "-m", _commit_message(bug))
                    self._log(job, f"pushing branch {branch}", bug)
                    pr = forge.open_pr(repo, branch, base, _pr_title(bug), _pr_body(job, bug))
                finally:
                    git(repo, "checkout", "--quiet", "--force", base)
            bug.pr = {"state": "open", "url": pr.url, "number": pr.number, "branch": branch}
            self._log(job, f"opened PR {pr.url}", bug)
        except (ForgeError, OSError, subprocess.SubprocessError, KeyError) as e:
            bug.pr = {"state": "failed", "error": str(e)[:600], "branch": branch}
            self._log(job, f"PR failed: {str(e)[:300]}", bug)
        finally:
            self.store.save(job)


def _has_identity(repo: Path) -> bool:
    r = subprocess.run(["git", "config", "user.email"], cwd=repo, capture_output=True, text=True)
    return bool(r.stdout.strip())


def _pr_title(bug: Bug) -> str:
    return bug.title if bug.source == "tests" else f"Fix #{bug.issue_number}: {bug.title}"


def _commit_message(bug: Bug) -> str:
    msg = _pr_title(bug)
    return msg + (f"\n\nFixes #{bug.issue_number}" if bug.issue_number else "")


def _pr_body(job: Job, bug: Bug) -> str:
    v = bug.verification or {}
    lines = ["## Summary", "", bug.summary or "Automated fix.", ""]
    if bug.issue_number:
        lines += [f"Fixes #{bug.issue_number}", ""]
    lines += ["## Verification", ""]
    if bug.tests:
        lines += ["Previously failing tests:", ""] + [
            f"- {'✅' if v.get('targets', {}).get(t) else '❌'} `{t}`" for t in bug.tests] + [""]
    else:
        lines += ["No failing test reproduces this issue, so the fix is **not verified by a test**. "
                  "Please review carefully.", ""]
    lines += [f"Full suite after the fix: {v.get('passed', '?')} passed, {v.get('failed', '?')} failed "
              f"(before: {job.baseline.get('passed', '?')} passed, {len(job.baseline.get('failed', []))} failed); "
              f"new failures: {len(v.get('new_failures', []))}.", "",
              "---", f"Generated by repo-fix-agent ({job.config.model}). Review before merging."]
    return "\n".join(lines)
