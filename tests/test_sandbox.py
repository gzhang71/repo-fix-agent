import os
import shutil
import subprocess
import sys
import time

import pytest

# allow running without `pip install -e .` (lead may replace with pyproject `pythonpath = ["src"]`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repo_fix_agent import sandbox as sb
from repo_fix_agent.sandbox import ExecResult, LocalSandbox, SandboxLimits, make_sandbox


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "orig"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    (src / "README.md").write_text("hello\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "junk.pyc").write_bytes(b"\0")
    (src / "node_modules").mkdir()
    (src / "node_modules" / "x.js").write_text("x")
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    return src


@pytest.fixture
def box(repo):
    s = LocalSandbox(str(repo))
    yield s
    s.close()


def _snapshot(path):
    return sorted((str(p.relative_to(path)), p.read_bytes() if p.is_file() else b"")
                  for p in path.rglob("*") if ".git" not in p.parts)


def test_copy_isolation(repo, box):
    before = _snapshot(repo)
    box.write_file("pkg/mod.py", "changed\n")
    box.write_file("new/file.txt", "x")
    assert box.exec("rm README.md").returncode == 0
    assert _snapshot(repo) == before
    assert not os.path.exists(os.path.join(box.root, "__pycache__"))
    assert not os.path.exists(os.path.join(box.root, "node_modules"))
    assert os.path.isabs(box.root)


def test_baseline_commit_and_clean_status(box):
    r = box.exec("git rev-parse HEAD && git rev-list --count HEAD && git status --porcelain")
    assert r.returncode == 0
    lines = r.output.split()
    assert lines[0] == box.baseline and lines[1] == "1" and len(lines) == 2
    box.write_file("pkg/mod.py", "def f():\n    return 2\n")
    diff = box.exec(f"git diff {box.baseline}").output
    assert "+    return 2" in diff


def test_exec_output_and_returncode(box):
    r = box.exec("echo out; echo err >&2; exit 3")
    assert isinstance(r, ExecResult)
    assert r.returncode == 3 and not r.timed_out
    assert "out" in r.output and "err" in r.output
    assert os.path.realpath(box.exec("pwd").output.strip()) == os.path.realpath(box.root)


def test_env_hygiene(box):
    out = box.exec("echo $PYTHONDONTWRITEBYTECODE $GIT_TERMINAL_PROMPT $HOME").output.split()
    assert out[:2] == ["1", "0"]
    assert out[2] != os.environ.get("HOME")


def test_timeout_kills_process_group(box, tmp_path):
    pidfile = tmp_path / "child.pid"
    start = time.monotonic()
    r = box.exec(f"echo started; sleep 30 & echo $! > {pidfile}; sleep 30", timeout=1)
    assert time.monotonic() - start < 8
    assert r.timed_out and r.returncode == 124
    assert "started" in r.output and "[timed out after 1s]" in r.output
    child = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("background child survived timeout")


def test_output_truncation_keeps_tail(repo):
    with LocalSandbox(str(repo), limits=SandboxLimits(max_output=100)) as s:
        r = s.exec("for i in $(seq 1 1000); do echo line$i; done")
    assert r.output.startswith("...[truncated]...\n")
    assert r.output.endswith("line1000\n")
    assert len(r.output) == len("...[truncated]...\n") + 100


def test_read_write_roundtrip(box):
    assert box.read_file("pkg/mod.py").startswith("def f()")
    box.write_file("a/b/c.txt", "héllo\n")
    assert box.read_file("a/b/c.txt") == "héllo\n"
    assert box.exec("cat a/b/c.txt").output == "héllo\n"
    with pytest.raises(FileNotFoundError):
        box.read_file("missing.txt")


@pytest.mark.parametrize("bad", ["../x", "pkg/../../x", "/etc/passwd", ""])
def test_path_escape_rejected(box, bad):
    with pytest.raises(ValueError):
        box.read_file(bad)
    with pytest.raises(ValueError):
        box.write_file(bad, "x")


def test_symlink_escape_rejected(box, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    os.symlink(outside, os.path.join(box.root, "link"))
    os.symlink(tmp_path, os.path.join(box.root, "dirlink"))
    with pytest.raises(ValueError):
        box.read_file("link")
    with pytest.raises(ValueError):
        box.write_file("dirlink/evil.txt", "x")
    assert not (tmp_path / "evil.txt").exists()


def test_python_resolves(box):
    r = box.exec("python -c 'print(1)'")
    assert r.returncode == 0 and r.output.strip() == "1"


def test_close_idempotent(repo):
    s = LocalSandbox(str(repo))
    tmp = os.path.dirname(s.root)
    assert os.path.isdir(tmp)
    s.close()
    s.close()
    assert not os.path.exists(tmp)


def test_context_manager(repo):
    with make_sandbox(str(repo), backend="local") as s:
        root = s.root
        assert s.exec("ls").returncode == 0
    assert not os.path.exists(root)


def test_auto_falls_back_to_local(repo, monkeypatch, caplog):
    monkeypatch.setattr(sb, "_docker_available", lambda: False)
    monkeypatch.setattr(sb, "_warned_fallback", False)  # warn-once flag may be set by earlier tests
    with make_sandbox(str(repo), backend="auto") as s:
        assert isinstance(s, LocalSandbox)
    assert any("falling back" in r.message for r in caplog.records)


def test_unknown_backend(repo):
    with pytest.raises(ValueError):
        make_sandbox(str(repo), backend="vm")


def test_git_url_clone_and_checkout(tmp_path, monkeypatch):
    """URL path: clone, checkout base_commit, drop history so later commits aren't reachable."""
    src = tmp_path / "upstream"
    src.mkdir()
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    (src / "a.txt").write_text("v1\n")
    subprocess.run(g + ["add", "-A"], cwd=src, check=True)
    subprocess.run(g + ["commit", "-qm", "v1"], cwd=src, check=True)
    v1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True,
                        text=True).stdout.strip()
    (src / "a.txt").write_text("v2\n")
    subprocess.run(g + ["commit", "-qam", "v2"], cwd=src, check=True)

    real_git = sb._git

    def offline_git(cwd, *args):  # redirect the network clone to the local upstream
        if args[0] == "clone":
            args = (*args[:-2], str(src), args[-1])
        return real_git(cwd, *args)

    monkeypatch.setattr(sb, "_git", offline_git)
    with LocalSandbox("https://example.invalid/r.git", base_commit=v1) as s:
        assert s.read_file("a.txt") == "v1\n"
        assert s.exec("git log --format=%H").output.split() == [s.baseline]


# ---- docker ----------------------------------------------------------------------------------
def test_docker_run_args_no_network():
    args = sb._docker_run_args("/host/repo", "img:1", SandboxLimits(memory_mb=512, max_procs=64))
    assert args[:2] == ["docker", "run"]
    i = args.index("--network")
    assert args[i + 1] == "none"
    assert args[args.index("--memory") + 1] == "512m"
    assert args[args.index("--pids-limit") + 1] == "64"
    assert args[args.index("-v") + 1] == "/host/repo:/workspace/repo"
    assert args[args.index("-w") + 1] == "/workspace/repo"
    assert args[-3:] == ["img:1", "sleep", "infinity"]


def test_docker_run_args_with_network():
    assert "--network" not in sb._docker_run_args("/r", "img", SandboxLimits(network=True))


def _docker_ok():
    return shutil.which("docker") is not None and sb._docker_available()


@pytest.mark.skipif(not _docker_ok(), reason="docker not available")
def test_docker_sandbox_end_to_end(repo):
    with make_sandbox(str(repo), backend="docker") as s:
        assert s.root == "/workspace/repo"
        r = s.exec("python -c 'print(1)'; pwd")
        assert r.returncode == 0 and r.output.split() == ["1", "/workspace/repo"]
        s.write_file("x.txt", "hi")
        assert s.exec("cat x.txt").output == "hi"
        assert s.read_file("pkg/mod.py").startswith("def f()")
        with pytest.raises(ValueError):
            s.read_file("../etc/passwd")
        start = time.monotonic()
        r = s.exec("sleep 30", timeout=2)
        assert r.timed_out and time.monotonic() - start < 30
        assert s.exec("echo alive").output.strip() == "alive"
        net = s.exec("python -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\"")
        assert net.returncode != 0
        s.close()
        s.close()
