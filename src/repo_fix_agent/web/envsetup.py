"""Per-repo environment images.

The agent's sandbox has no network, so a repo's dependencies must be installed beforehand. We do
that once per job: copy the checkout (no .git) to a temp dir, mount it at the sandbox path, run the
setup command *with* network inside the base sandbox image, and `docker commit` the result. Every
agent episode and verification run then uses that image with --network none. Editable installs
point at /workspace/repo, which is where each sandbox mounts its own copy of the repo.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from ..sandbox import CONTAINER_ROOT, IGNORE, default_image

DEFAULT_SETUP = r"""
set -u
pip install -q --no-cache-dir --upgrade pip >/dev/null 2>&1 || true
for f in requirements.txt requirements-dev.txt requirements-test.txt requirements_dev.txt \
         dev-requirements.txt test-requirements.txt requirements/test.txt requirements/dev.txt; do
  if [ -f "$f" ]; then echo "+ pip install -r $f"; pip install -q --no-cache-dir -r "$f" || exit 1; fi
done
if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then
  echo "+ pip install -e .[test|dev|tests]"
  pip install -q --no-cache-dir -e ".[test]" 2>/dev/null || pip install -q --no-cache-dir -e ".[tests]" 2>/dev/null \
    || pip install -q --no-cache-dir -e ".[dev]" 2>/dev/null || pip install -q --no-cache-dir -e . || exit 1
fi
python -m pytest --version >/dev/null 2>&1 || pip install -q --no-cache-dir pytest
"""


class SetupError(RuntimeError):
    pass


def build_env_image(repo: Path, tag: str, setup_cmd: str | None = None, base_image: str | None = None,
                    timeout: int = 1800) -> str:
    """Return `tag` after installing the repo's dependencies into a copy of the base image."""
    base_image = base_image or default_image()
    name = f"repofix-setup-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="repofix-setup-", ignore_cleanup_errors=True) as tmp:  # root-owned build files
        work = Path(tmp) / "repo"
        shutil.copytree(repo, work, symlinks=True, ignore=shutil.ignore_patterns(*IGNORE))
        run = ["docker", "run", "--name", name, "-v", f"{work}:{CONTAINER_ROOT}", "-w", CONTAINER_ROOT,
               "--memory", "4096m", "--pids-limit", "512", "--security-opt", "no-new-privileges",
               base_image, "bash", "-c", setup_cmd or DEFAULT_SETUP]
        try:
            r = subprocess.run(run, capture_output=True, text=True, timeout=timeout)
            log = (r.stdout + r.stderr)[-4000:]
            if r.returncode != 0:
                raise SetupError(f"setup command failed (exit {r.returncode}):\n{log}")
            # sandboxes run as the host uid: keep HOME world-writable, never root's home
            c = subprocess.run(["docker", "commit", "--change", "ENV HOME=/tmp", name, tag],
                               capture_output=True, text=True, timeout=300)
            if c.returncode != 0:
                raise SetupError(f"docker commit failed: {c.stderr.strip()}")
        except subprocess.TimeoutExpired:
            raise SetupError(f"setup timed out after {timeout}s") from None
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    return tag


def remove_image(tag: str) -> None:
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)
