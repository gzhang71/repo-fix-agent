"""Where fixes go: a git host that can receive branches and open pull requests.

GitHubForge talks to the GitHub REST API with a token from $GITHUB_TOKEN, falling back to the
logged-in GitHub CLI (`gh auth token`). Pushes go through a GIT_ASKPASS helper so the token never
appears in a command line, a remote URL, a log line, or the browser. When the token can't push to
the repo, the fix branch goes to a fork and the PR is opened from there.

LocalForge is for non-GitHub URLs (and tests): it can push branches to the remote but cannot open
PRs, so the UI offers the patch for download instead.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_GH_URL = re.compile(r"^(?:https?://github\.com/|git@github\.com:|ssh://git@github\.com/)"
                     r"(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?/?$")


@dataclass
class RepoRef:
    url: str
    owner: str | None = None
    name: str | None = None

    @property
    def is_github(self) -> bool:
        return self.owner is not None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}" if self.is_github else self.url


def parse_repo_url(url: str) -> RepoRef:
    url = url.strip()
    m = _GH_URL.match(url)
    if m:
        return RepoRef(f"https://github.com/{m['owner']}/{m['repo']}.git", m["owner"], m["repo"])
    if not (url.startswith(("https://", "http://", "git@", "ssh://", "file://")) or os.path.isdir(url)):
        raise ValueError(f"not a git URL: {url!r}")
    return RepoRef(url)


@dataclass
class PullRequest:
    number: int | None
    url: str
    branch: str


class ForgeError(RuntimeError):
    pass


def github_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok.strip()
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def _askpass(dirpath: Path) -> Path:
    p = dirpath / "git-askpass.sh"
    p.write_text('#!/bin/sh\ncase "$1" in\n  Username*) echo x-access-token ;;\n'
                 '  *) printf "%s\\n" "$REPOFIX_GIT_TOKEN" ;;\nesac\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def git(cwd: str | Path, *args: str, token: str | None = None, askpass_dir: Path | None = None,
        timeout: int = 600) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token and askpass_dir is not None:
        env.update(GIT_ASKPASS=str(_askpass(askpass_dir)), REPOFIX_GIT_TOKEN=token)
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        if token:
            msg = msg.replace(token, "***")
        raise ForgeError(f"git {args[0]} failed: {msg[-800:]}")
    return r.stdout


class LocalForge:
    """Any git remote: clone and push branches, no pull requests."""
    can_open_prs = False

    def __init__(self, ref: RepoRef, workdir: Path):
        self.ref, self.workdir = ref, workdir

    def clone(self, dest: Path, branch: str | None = None) -> str:
        args = ["clone", "--quiet"] + (["--branch", branch] if branch else []) + ["--", self.ref.url, str(dest)]
        git(self.workdir, *args)
        return git(dest, "rev-parse", "--abbrev-ref", "HEAD").strip()

    def push(self, repo: Path, branch: str) -> str:
        git(repo, "push", "--quiet", "origin", f"{branch}:{branch}")
        return "origin"

    def open_pr(self, repo: Path, branch: str, base: str, title: str, body: str) -> PullRequest:
        raise ForgeError("this remote is not on GitHub; download the patch instead")


class GitHubForge(LocalForge):
    can_open_prs = True
    API = "https://api.github.com"

    def __init__(self, ref: RepoRef, workdir: Path, token: str | None = None, opener=None):
        super().__init__(ref, workdir)
        self.token = token if token is not None else github_token()
        self._open = opener or urllib.request.urlopen
        self._push_target: tuple[str, str] | None = None   # (owner, repo) branches are pushed to

    # -- REST -----------------------------------------------------------------------------------
    def api(self, method: str, path: str, body: dict | None = None) -> dict | list:
        if not self.token:
            raise ForgeError("no GitHub token: set GITHUB_TOKEN or run `gh auth login`")
        req = urllib.request.Request(
            self.API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "repo-fix-agent"})
        try:
            with self._open(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise ForgeError(f"GitHub {method} {path} -> {e.code}: {detail}") from None
        return json.loads(raw) if raw else {}

    def repo_info(self) -> dict:
        return self.api("GET", f"/repos/{self.ref.owner}/{self.ref.name}")

    def bug_issues(self, label: str = "bug", limit: int = 20) -> list[dict]:
        items = self.api("GET", f"/repos/{self.ref.owner}/{self.ref.name}/issues"
                                f"?state=open&labels={label}&per_page={min(100, limit)}")
        return [i for i in items if "pull_request" not in i][:limit]

    # -- git ------------------------------------------------------------------------------------
    def clone(self, dest: Path, branch: str | None = None) -> str:
        args = ["clone", "--quiet"] + (["--branch", branch] if branch else []) + ["--", self.ref.url, str(dest)]
        git(self.workdir, *args, token=self.token, askpass_dir=self.workdir)
        return git(dest, "rev-parse", "--abbrev-ref", "HEAD").strip()

    def _target(self) -> tuple[str, str]:
        if self._push_target is None:
            info = self.repo_info()
            if (info.get("permissions") or {}).get("push"):
                self._push_target = (self.ref.owner, self.ref.name)
            else:
                fork = self.api("POST", f"/repos/{self.ref.owner}/{self.ref.name}/forks", {})
                owner, name = fork["owner"]["login"], fork["name"]
                for _ in range(30):   # forks are created asynchronously
                    try:
                        self.api("GET", f"/repos/{owner}/{name}")
                        break
                    except ForgeError:
                        time.sleep(2)
                self._push_target = (owner, name)
        return self._push_target

    def push(self, repo: Path, branch: str) -> str:
        owner, name = self._target()
        git(repo, "push", "--quiet", f"https://github.com/{owner}/{name}.git", f"{branch}:{branch}",
            token=self.token, askpass_dir=self.workdir)
        return f"{owner}/{name}"

    def open_pr(self, repo: Path, branch: str, base: str, title: str, body: str) -> PullRequest:
        self.push(repo, branch)
        owner, _ = self._target()
        head = branch if owner == self.ref.owner else f"{owner}:{branch}"
        pr = self.api("POST", f"/repos/{self.ref.owner}/{self.ref.name}/pulls",
                      {"title": title, "head": head, "base": base, "body": body,
                       "maintainer_can_modify": True})
        return PullRequest(pr.get("number"), pr.get("html_url", ""), branch)


def make_forge(ref: RepoRef, workdir: Path) -> LocalForge:
    return GitHubForge(ref, workdir) if ref.is_github else LocalForge(ref, workdir)
