"""Execution sandboxes for agent tool calls.

Security model
--------------
Every sandbox works on a private *copy* of the target repo (local dir copied, or git URL cloned
and checked out at ``base_commit``). The copy is re-initialised as a fresh git repo with a single
``baseline`` commit, so the original checkout is never modified, future commits of a cloned repo
are not reachable from inside the sandbox, and ``git diff <baseline>`` yields the agent's patch.

* ``DockerSandbox`` (production): a long-lived container with ``--network none``, memory, pids
  and CPU limits, ``no-new-privileges``; commands run via ``docker exec ... bash -lc`` and the
  timeout is enforced host-side (on timeout every process in the container except PID 1 is
  killed). File I/O goes through the host-side bind-mounted copy with path-escape checks.
* ``LocalSandbox`` (development only): a temp-dir copy, commands run as the *current user* with
  rlimits (CPU always; address space and process count on Linux only) and a scrubbed HOME. It
  is NOT a security boundary: it cannot block network access, and a command may read or write
  anything the user can (path checks only guard ``read_file``/``write_file``). Use docker for
  untrusted code.

All subprocesses are spawned with argv lists; the agent's command is only ever interpreted by a
shell running *inside* the sandbox (cwd = repo root).
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

IGNORE = (".git", "__pycache__", ".pytest_cache", ".venv", "node_modules")
GIT_ID = ["-c", "user.name=agent", "-c", "user.email=agent@local", "-c", "commit.gpgsign=false"]
TRUNC_PREFIX = "...[truncated]...\n"


@dataclass
class ExecResult:
    returncode: int
    output: str
    timed_out: bool = False


@dataclass
class SandboxLimits:
    cpu_seconds: int = 300
    memory_mb: int = 4096
    max_output: int = 8000
    network: bool = False
    max_procs: int = 256


class Sandbox(Protocol):
    root: str
    baseline: str

    def exec(self, command: str, timeout: int = 120) -> ExecResult: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Sandbox": ...
    def __exit__(self, *exc) -> None: ...


# ---- helpers ---------------------------------------------------------------------------------
def _git(cwd: str, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def _prepare_repo(repo: str, base_commit: str | None, dest: str) -> str:
    """Materialise `repo` at `dest` as a fresh git repo with one baseline commit; return its sha."""
    if repo.startswith(("http://", "https://", "git@")):
        _git(os.path.dirname(dest), "clone", "--quiet", "--", repo, dest)
        if base_commit:
            _git(dest, "checkout", "--quiet", base_commit)
        shutil.rmtree(os.path.join(dest, ".git"))  # drop history (incl. future commits)
    else:
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"repo not found: {repo}")
        shutil.copytree(repo, dest, symlinks=True, ignore=shutil.ignore_patterns(*IGNORE))
    _git(dest, "init", "--quiet")
    _git(dest, "add", "-A")
    _git(dest, *GIT_ID, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", "baseline")
    return _git(dest, "rev-parse", "HEAD").strip()


def _safe_path(root: str, path: str) -> str:
    if not path or os.path.isabs(path):
        raise ValueError(f"path must be relative to the repo root: {path!r}")
    real_root = os.path.realpath(root)
    p = os.path.realpath(os.path.join(real_root, path))
    if os.path.commonpath([real_root, p]) != real_root:
        raise ValueError(f"path escapes repository: {path!r}")
    return p


def _truncate(out: str, limit: int) -> str:
    return out if len(out) <= limit else TRUNC_PREFIX + out[-limit:]


def _run(argv: list[str], timeout: int, max_output: int, on_timeout=None, **popen_kw) -> ExecResult:
    """Run argv with combined output; on timeout kill its whole process group. Never raises."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, **popen_kw)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return ExecResult(proc.returncode, _truncate(out.decode(errors="replace"), max_output))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if on_timeout:
            on_timeout()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # a descendant escaped the group but holds the pipe
            proc.kill()
            out = b""
        text = out.decode(errors="replace") if out else ""
        return ExecResult(124, _truncate(text, max_output) + f"\n[timed out after {timeout}s]", True)


class _Base:
    root: str
    baseline: str
    _files_root: str  # host path backing read_file/write_file

    def read_file(self, path: str) -> str:
        with open(_safe_path(self._files_root, path), encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        p = _safe_path(self._files_root, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---- local backend ---------------------------------------------------------------------------
_warned_network = False


class LocalSandbox(_Base):
    """Temp-dir sandbox for development. Not a security boundary (see module docstring)."""

    def __init__(self, repo: str, base_commit: str | None = None, limits: SandboxLimits | None = None):
        global _warned_network
        self.limits = limits or SandboxLimits()
        if not self.limits.network and not _warned_network:
            log.warning("local backend cannot disable network; use docker in production")
            _warned_network = True
        self._tmp: str | None = tempfile.mkdtemp(prefix="repofix-")
        try:
            self.root = self._files_root = os.path.join(self._tmp, "repo")
            self.baseline = _prepare_repo(str(repo), base_commit, self.root)
            bindir, home = os.path.join(self._tmp, "bin"), os.path.join(self._tmp, "home")
            os.makedirs(bindir)
            os.makedirs(home)
            os.symlink(sys.executable, os.path.join(bindir, "python"))
            self.env = {**os.environ, "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
                        "HOME": home, "PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}
        except BaseException:
            self.close()
            raise

    def _preexec(self) -> None:  # runs in the child between fork and exec
        import resource

        def lower(res: int, value: int) -> None:
            try:
                _, hard = resource.getrlimit(res)
                if hard != resource.RLIM_INFINITY:
                    value = min(value, hard)
                resource.setrlimit(res, (value, value if hard == resource.RLIM_INFINITY else hard))
            except (ValueError, OSError):
                pass

        lower(resource.RLIMIT_CPU, self.limits.cpu_seconds)
        if sys.platform.startswith("linux"):  # macOS: RLIMIT_AS unreliable; NPROC is per-user
            lower(resource.RLIMIT_AS, self.limits.memory_mb * 1024 * 1024)
            lower(resource.RLIMIT_NPROC, self.limits.max_procs)

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        if self._tmp is None:
            raise RuntimeError("sandbox is closed")
        return _run(["/bin/bash", "-c", command], timeout, self.limits.max_output,
                    cwd=self.root, env=self.env, preexec_fn=self._preexec)

    def close(self) -> None:
        tmp, self._tmp = getattr(self, "_tmp", None), None
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ---- docker backend --------------------------------------------------------------------------
CONTAINER_ROOT = "/workspace/repo"


def _docker_run_args(host_repo: str, image: str, limits: SandboxLimits) -> list[str]:
    args = ["docker", "run", "-d", "--rm"]
    if not limits.network:
        args += ["--network", "none"]
    args += ["--memory", f"{limits.memory_mb}m", "--pids-limit", str(limits.max_procs),
             "--cpus", "2", "--ulimit", f"cpu={limits.cpu_seconds}",
             "--security-opt", "no-new-privileges",
             "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "GIT_TERMINAL_PROMPT=0",
             "-v", f"{host_repo}:{CONTAINER_ROOT}", "-w", CONTAINER_ROOT,
             image, "sleep", "infinity"]
    return args


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox(_Base):
    root = CONTAINER_ROOT

    def __init__(self, repo: str, base_commit: str | None = None,
                 limits: SandboxLimits | None = None, image: str = "python:3.12"):
        self.limits = limits or SandboxLimits()
        self.container: str | None = None
        self._tmp: str | None = tempfile.mkdtemp(prefix="repofix-")
        try:
            self._files_root = os.path.join(self._tmp, "repo")
            self.baseline = _prepare_repo(str(repo), base_commit, self._files_root)
            r = subprocess.run(_docker_run_args(self._files_root, image, self.limits),
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
            self.container = r.stdout.strip()
        except BaseException:
            self.close()
            raise

    def _kill_all(self) -> None:
        # kill(-1) signals every process but PID 1 (`sleep infinity`) and the caller itself.
        try:
            subprocess.run(["docker", "exec", self.container, "sh", "-c", "kill -9 -1"],
                           capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            log.warning("failed to kill processes in container %s after timeout", self.container)

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        if self.container is None:
            raise RuntimeError("sandbox is closed")
        return _run(["docker", "exec", self.container, "bash", "-lc", command], timeout,
                    self.limits.max_output, on_timeout=self._kill_all)

    def close(self) -> None:
        cid, self.container = getattr(self, "container", None), None
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
        tmp, self._tmp = getattr(self, "_tmp", None), None
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ---- factory ---------------------------------------------------------------------------------
def make_sandbox(repo: str, base_commit: str | None = None, backend: str = "auto",
                 limits: SandboxLimits | None = None, image: str = "python:3.12") -> Sandbox:
    if backend == "auto":
        backend = "docker" if _docker_available() else "local"
        if backend == "local":
            log.warning("docker unavailable; falling back to the local (development-only) sandbox")
    if backend == "docker":
        return DockerSandbox(repo, base_commit, limits, image)
    if backend == "local":
        return LocalSandbox(repo, base_commit, limits)
    raise ValueError(f"unknown sandbox backend: {backend!r}")
