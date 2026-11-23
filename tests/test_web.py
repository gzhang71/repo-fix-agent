"""Web app: URL parsing, pytest parsing, full job pipeline against a local git remote, HTTP API."""
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action
from repo_fix_agent.web import forge as F
from repo_fix_agent.web import jobs as J
from repo_fix_agent.web.server import serve

G = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


# ---- parsing ---------------------------------------------------------------------------------
@pytest.mark.parametrize("url,slug", [
    ("https://github.com/psf/requests", "psf/requests"),
    ("https://github.com/psf/requests.git", "psf/requests"),
    ("git@github.com:psf/requests.git", "psf/requests"),
    ("https://github.com/a-b/c.d/", "a-b/c.d"),
])
def test_parse_github_urls(url, slug):
    ref = F.parse_repo_url(url)
    assert ref.is_github and ref.slug == slug and ref.url == f"https://github.com/{slug}.git"


def test_parse_other_urls():
    assert not F.parse_repo_url("https://gitlab.com/a/b.git").is_github
    with pytest.raises(ValueError):
        F.parse_repo_url("rm -rf /")


PYTEST_OUT = """\
F.E
=================================== FAILURES ===================================
__________________________________ test_add ___________________________________
    def test_add():
>       assert add(2, 2) == 4
E       assert 5 == 4
tests/test_calc.py:4: AssertionError
_____________________________ TestX.test_method ______________________________
E   ValueError: boom
==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_broken.py _________________
ImportError: nope
=========================== short test summary info ============================
PASSED tests/test_calc.py::test_sub
FAILED tests/test_calc.py::test_add - assert 5 == 4
FAILED tests/test_calc.py::TestX::test_method - ValueError: boom
ERROR tests/test_broken.py - ImportError: nope
SKIPPED [1] tests/test_calc.py:9: why
"""


def test_parse_pytest_and_sections():
    r = J.parse_pytest(PYTEST_OUT, 1)
    assert r.passed == {"tests/test_calc.py::test_sub"}
    assert r.failed == {"tests/test_calc.py::test_add", "tests/test_calc.py::TestX::test_method", "tests/test_broken.py"}
    s = J.failure_sections(PYTEST_OUT)
    assert "assert 5 == 4" in J._section_for("tests/test_calc.py::test_add", s)
    assert "boom" in J._section_for("tests/test_calc.py::TestX::test_method", s)
    assert "ImportError" in J._section_for("tests/test_broken.py", s)


def test_pytest_cmd_quotes_ids():
    assert J.pytest_cmd("python -m pytest -q", ["t.py::a b"]).endswith("'t.py::a b'")
    assert "-rA" in J.pytest_cmd("python -m pytest -q")


def test_job_config_clamps():
    c = J.JobConfig.from_dict({"repo_url": "x", "max_bugs": 999, "workers": 0, "sources": ["bogus"], "branch": ""})
    assert (c.max_bugs, c.workers, c.sources, c.branch) == (50, 1, ["tests"], None)


# ---- full pipeline on a local remote ---------------------------------------------------------
@pytest.fixture
def remote(tmp_path):
    return make_remote(tmp_path)


def make_remote(tmp_path):
    """Bare 'origin' with two independent bugs (two failing test files) and one passing test."""
    src = tmp_path / "src"
    (src / "calc").mkdir(parents=True)
    (src / "tests").mkdir()
    (src / "calc" / "__init__.py").write_text("")
    (src / "calc" / "ops.py").write_text("def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n")
    (src / "calc" / "text.py").write_text("def shout(s):\n    return s.lower() + '!'\n")
    (src / "tests" / "test_ops.py").write_text(
        "from calc.ops import add, sub\n\ndef test_add():\n    assert add(2, 2) == 4\n\n"
        "def test_sub():\n    assert sub(3, 1) == 2\n")
    (src / "tests" / "test_text.py").write_text(
        "from calc.text import shout\n\ndef test_shout():\n    assert shout('hi') == 'HI!'\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=src, check=True)
    subprocess.run(G + ["add", "-A"], cwd=src, check=True)
    subprocess.run(G + ["commit", "-qm", "init"], cwd=src, check=True)
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)], check=True)
    return bare


FIXES = {
    "test_ops.py": ("calc/ops.py", "return a + b + 1", "return a + b"),
    "test_text.py": ("calc/text.py", "s.lower()", "s.upper()"),
}


class FixingPolicy(ScriptedPolicy):
    """Reads the task, applies the matching fix (or a wrong one for `sabotage`), runs tests, submits."""
    def __init__(self, sabotage=()):
        super().__init__([])
        self.sabotage = set(sabotage)

    def reset(self, system, task):
        steps = []
        for test_file, (path, old, new) in FIXES.items():
            if test_file in task:
                if test_file in self.sabotage:
                    path, old, new = "calc/ops.py", "return a - b", "return b - a"   # breaks test_sub
                steps += [("look", Action("view_file", {"path": path})),
                          ("fix", Action("str_replace", {"path": path, "old_str": old, "new_str": new}))]
        steps += [("verify", Action("bash", {"command": "python -m pytest -q -p no:cacheprovider"})),
                  ("done", Action("submit", {}))]
        self._steps = steps
        self._i = 0


class RecordingForge(F.LocalForge):
    can_open_prs = True
    opened = []

    def open_pr(self, repo, branch, base, title, body):
        self.push(repo, branch)
        RecordingForge.opened.append({"branch": branch, "base": base, "title": title, "body": body})
        return F.PullRequest(len(RecordingForge.opened), f"https://example.test/pr/{len(RecordingForge.opened)}", branch)


def _runner(tmp_path, sabotage=()):
    RecordingForge.opened = []
    return J.JobRunner(J.JobStore(tmp_path / "jobs"), backend="local",
                       policy_factory=lambda model: FixingPolicy(sabotage),
                       forge_factory=lambda ref, wd: RecordingForge(ref, wd))


def _wait(job, until=("done", "failed", "cancelled"), timeout=120):
    t0 = time.time()
    while job.status not in until:
        assert time.time() - t0 < timeout, f"stuck in {job.status}: {job.events[-5:]}"
        time.sleep(0.1)


def test_job_finds_fixes_verifies_and_opens_prs(tmp_path, remote):
    runner = _runner(tmp_path)
    job = runner.submit(J.JobConfig(repo_url=str(remote), workers=2))
    _wait(job)
    assert job.status == "done", job.error
    assert job.repo["base_branch"] == "main" and job.baseline["passed"] == 1
    assert sorted(job.baseline["failed"]) == ["tests/test_ops.py::test_add", "tests/test_text.py::test_shout"]
    assert [b.tests for b in job.bugs] == [["tests/test_ops.py::test_add"], ["tests/test_text.py::test_shout"]]
    assert all(b.status == "verified" for b in job.bugs), [(b.status, b.error) for b in job.bugs]
    assert "tests/" not in "".join(b.patch for b in job.bugs)   # fixes never touch tests
    assert any(e["msg"].startswith("step 2: str_replace") for e in job.events)

    for b in job.bugs:
        runner.open_pr(job, b, wait=True)
    assert all(b.pr["state"] == "open" for b in job.bugs), [b.pr for b in job.bugs]
    assert len(RecordingForge.opened) == 2 and RecordingForge.opened[0]["base"] == "main"
    assert "✅ `tests/test_ops.py::test_add`" in RecordingForge.opened[0]["body"]
    # each PR branch landed on the remote, based on the original commit, with only its own fix
    for b in job.bugs:
        files = subprocess.run(["git", "diff", "--name-only", "main", b.pr["branch"]], cwd=remote,
                               capture_output=True, text=True, check=True).stdout.split()
        assert files == [FIXES[b.tests[0].split("/")[1].split("::")[0]][0]]
    # the workspace is back on the base branch, clean
    ws = runner.store.dir(job) / "repo"
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True).stdout == ""


def test_regression_is_caught_and_not_auto_pr(tmp_path, remote):
    runner = _runner(tmp_path, sabotage={"test_text.py"})
    job = runner.submit(J.JobConfig(repo_url=str(remote), auto_pr=True))
    _wait(job)
    st = {b.tests[0].split("::")[0]: b for b in job.bugs}
    assert st["tests/test_ops.py"].status == "verified" and st["tests/test_ops.py"].pr["state"] == "open"
    bad = st["tests/test_text.py"]
    assert bad.status == "regressed" and "tests/test_ops.py::test_sub" in bad.verification["new_failures"]
    assert bad.pr is None and len(RecordingForge.opened) == 1   # auto_pr only for verified fixes


def test_clean_repo_reports_no_bugs(tmp_path, remote):
    work = tmp_path / "w"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True)
    (work / "calc" / "ops.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    (work / "calc" / "text.py").write_text("def shout(s):\n    return s.upper() + '!'\n")
    subprocess.run(G + ["commit", "-qam", "fix"], cwd=work, check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=work, check=True)
    job = _runner(tmp_path).submit(J.JobConfig(repo_url=str(remote)))
    _wait(job)
    assert job.status == "done" and job.bugs == [] and "no bugs found" in job.events[-1]["msg"]


def test_bad_model_fails_fast(tmp_path, remote):
    def boom(model):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    runner = J.JobRunner(J.JobStore(tmp_path / "jobs"), backend="local", policy_factory=boom)
    job = runner.submit(J.JobConfig(repo_url=str(remote)))
    _wait(job)
    assert job.status == "failed" and "ANTHROPIC_API_KEY" in job.error


def test_store_reloads_and_marks_interrupted(tmp_path, remote):
    store = J.JobStore(tmp_path / "jobs")
    job = store.create(J.JobConfig(repo_url=str(remote)))
    job.status = "fixing"
    job.bugs = [J.Bug("t1", "x", "tests", "ps", status="fixing")]
    store.save(job)
    again = J.JobStore(tmp_path / "jobs").get(job.id)
    assert again.status == "failed" and again.bugs[0].status == "cancelled"


# ---- GitHub forge (fake HTTP) ----------------------------------------------------------------
class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=0):
        key = (req.get_method(), req.full_url.replace(F.GitHubForge.API, ""))
        self.calls.append((key, json.loads(req.data) if req.data else None, req.headers))
        body = self.routes[key]
        resp = type("R", (), {"read": lambda s: json.dumps(body).encode(), "__enter__": lambda s: s,
                              "__exit__": lambda s, *a: None})()
        return resp


def test_github_forge_forks_when_no_push(tmp_path, monkeypatch):
    http = FakeHTTP({
        ("GET", "/repos/up/proj"): {"permissions": {"push": False}},
        ("POST", "/repos/up/proj/forks"): {"owner": {"login": "me"}, "name": "proj"},
        ("GET", "/repos/me/proj"): {},
        ("POST", "/repos/up/proj/pulls"): {"number": 7, "html_url": "https://github.com/up/proj/pull/7"},
    })
    fg = F.GitHubForge(F.parse_repo_url("https://github.com/up/proj"), tmp_path, token="tok123", opener=http)
    pushed = []
    monkeypatch.setattr(F, "git", lambda cwd, *args, **kw: pushed.append(args) or "")
    pr = fg.open_pr(tmp_path, "repo-fix/x", "main", "Fix", "body")
    assert pr.number == 7 and pr.url.endswith("/pull/7")
    assert pushed[0][2] == "https://github.com/me/proj.git"            # pushed to the fork
    post = next(c for c in http.calls if c[0] == ("POST", "/repos/up/proj/pulls"))
    assert post[1]["head"] == "me:repo-fix/x" and post[1]["base"] == "main"
    assert post[2]["Authorization"] == "Bearer tok123"
    assert "tok123" not in json.dumps([c[1] for c in http.calls])


def test_git_errors_redact_token(tmp_path):
    with pytest.raises(F.ForgeError) as e:
        F.git(tmp_path, "push", "https://x-access-token:SECRET@127.0.0.1:9/none.git", "main", token="SECRET",
              askpass_dir=tmp_path)
    assert "SECRET" not in str(e.value)


# ---- HTTP API ----------------------------------------------------------------------------------
@pytest.fixture
def server(tmp_path):
    runner = _runner(tmp_path)
    httpd = serve("127.0.0.1", 0, runner=runner)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", runner
    httpd.shutdown()


def _req(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_http_flow(server, remote):
    base, runner = server
    code, html, _ = _req(base + "/")
    assert code == 200 and b"<title>Repo Fix</title>" in html
    code, cfg, _ = _req(base + "/api/config")
    assert code == 200 and json.loads(cfg)["backend"] == "local"
    assert _req(base + "/api/jobs", "POST", {})[0] == 400
    assert _req(base + "/api/jobs", "POST", {"repo_url": "not a url"})[0] == 400
    code, body, _ = _req(base + "/api/jobs", "POST", {"repo_url": str(remote)})
    assert code == 201
    jid = json.loads(body)["id"]
    _wait(runner.store.get(jid))
    code, body, _ = _req(f"{base}/api/jobs/{jid}?since=3")
    job = json.loads(body)
    assert job["status"] == "done" and all(e["seq"] > 3 for e in job["events"])
    bug = job["bugs"][0]["id"]
    code, patch, hdrs = _req(f"{base}/api/jobs/{jid}/bugs/{bug}/patch")
    assert code == 200 and patch.startswith(b"diff --git") and "attachment" in hdrs["Content-Disposition"]
    # cross-site POST refused (CSRF)
    assert _req(f"{base}/api/jobs/{jid}/prs", "POST", {}, {"Origin": "https://evil.example"})[0] == 403
    code, body, _ = _req(f"{base}/api/jobs/{jid}/prs", "POST", {})
    assert code == 202 and len(json.loads(body)["opening"]) == 2
    t0 = time.time()
    while not all((b.pr or {}).get("state") == "open" for b in runner.store.get(jid).bugs):
        assert time.time() - t0 < 30
        time.sleep(0.1)
    summaries = json.loads(_req(base + "/api/jobs")[1])
    assert summaries[0]["n_prs"] == 2 and "bugs" not in summaries[0]
    assert _req(f"{base}/api/jobs/nope")[0] == 404


def test_credentials_checked_before_cloning(monkeypatch):
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        J.check_model_credentials("claude-opus-5-5")
    with pytest.raises(RuntimeError, match="openai|OPENAI_API_KEY"):
        J.check_model_credentials("gpt-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    J.check_model_credentials("claude-opus-5-5")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        J.check_model_credentials("student")   # hosted student needs its key
