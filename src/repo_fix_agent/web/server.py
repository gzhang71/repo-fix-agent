"""Local web UI for repo-fix: paste a repo URL, the agent fixes its bugs, you open PRs.

    repo-fix serve [--port 8765] [--host 127.0.0.1] [--jobs-dir runs/web]

Standard library only (http.server); the page polls the JSON API. It binds to localhost by
default: anyone who can reach it can make the server clone repos and push branches with your
GitHub credentials, so don't expose it without putting auth in front.

API
  GET  /api/config                         models, backend, GitHub auth available
  GET  /api/jobs                           job summaries
  POST /api/jobs            {repo_url,...} start a job (JobConfig fields)
  GET  /api/jobs/<id>?since=<seq>          job state + events after seq
  POST /api/jobs/<id>/cancel
  POST /api/jobs/<id>/bugs/<bug>/pr        open a PR for one fix
  POST /api/jobs/<id>/prs                  open PRs for every verified fix
  GET  /api/jobs/<id>/bugs/<bug>/patch     the fix as a .patch file
"""
from __future__ import annotations

import json
import logging
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .forge import github_token
from .jobs import DEFAULT_TEST_CMD, JobConfig, JobRunner, JobStore

STATIC = Path(__file__).parent / "static"
MODELS = [("claude-opus-5-5", "Claude Opus 5.5 (teacher)"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
          ("gpt-5", "GPT-5"), ("student", "Student (vLLM/SGLang, configs/models.yaml)")]
log = logging.getLogger(__name__)


def make_handler(runner: JobRunner):
    store = runner.store

    class Handler(BaseHTTPRequestHandler):
        server_version = "repo-fix"

        def log_message(self, fmt, *args):  # quieter than the default stderr spam
            log.debug("%s %s", self.address_string(), fmt % args)

        # -- helpers --------------------------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=list).encode(), "application/json")

        def _err(self, code: int, msg: str) -> None:
            self._json({"error": msg}, code)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _job_bug(self, job_id: str, bug_id: str | None = None):
            job = store.get(job_id)
            if job is None:
                return None, None
            bug = next((b for b in job.bugs if b.id == bug_id), None) if bug_id else None
            return job, bug

        def _same_origin(self) -> bool:
            # Block cross-site form posts / fetches from other pages (CSRF): browsers send Origin.
            origin = self.headers.get("Origin")
            return origin is None or urlparse(origin).netloc == self.headers.get("Host")

        # -- routes ---------------------------------------------------------------------------
        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            path = u.path.rstrip("/") or "/"
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/config":
                return self._json({"models": [{"id": m, "label": l} for m, l in MODELS],
                                   "default_test_cmd": DEFAULT_TEST_CMD, "backend": runner.backend,
                                   "github_auth": bool(github_token())})
            if path == "/api/jobs":
                return self._json([j.to_dict(full=False) for j in store.list()])
            m = re.fullmatch(r"/api/jobs/([\w-]+)", path)
            if m:
                job, _ = self._job_bug(m.group(1))
                if job is None:
                    return self._err(404, "no such job")
                since = int(parse_qs(u.query).get("since", ["0"])[0] or 0)
                return self._json(job.to_dict(since=since))
            m = re.fullmatch(r"/api/jobs/([\w-]+)/bugs/([\w-]+)/patch", path)
            if m:
                job, bug = self._job_bug(m.group(1), m.group(2))
                if bug is None or not bug.patch:
                    return self._err(404, "no patch")
                return self._send(200, bug.patch.encode(), "text/x-diff; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{job.id}-{bug.id}.patch"'})
            return self._err(404, "not found")

        def do_POST(self):  # noqa: N802
            if not self._same_origin():
                return self._err(403, "cross-origin request refused")
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as e:
                return self._err(400, f"bad request: {e}")
            if path == "/api/jobs":
                if not body.get("repo_url"):
                    return self._err(400, "repo_url is required")
                try:
                    job = runner.submit(JobConfig.from_dict(body))
                except (ValueError, TypeError) as e:
                    return self._err(400, str(e))
                return self._json(job.to_dict(), 201)
            m = re.fullmatch(r"/api/jobs/([\w-]+)/cancel", path)
            if m:
                job, _ = self._job_bug(m.group(1))
                if job is None:
                    return self._err(404, "no such job")
                runner.cancel(job)
                return self._json({"ok": True})
            m = re.fullmatch(r"/api/jobs/([\w-]+)/bugs/([\w-]+)/pr", path)
            if m:
                job, bug = self._job_bug(m.group(1), m.group(2))
                if bug is None:
                    return self._err(404, "no such bug")
                if not job.repo.get("can_open_prs"):
                    return self._err(400, "this repo is not on GitHub; download the patch instead")
                try:
                    runner.open_pr(job, bug)
                except ValueError as e:
                    return self._err(400, str(e))
                return self._json({"ok": True}, 202)
            m = re.fullmatch(r"/api/jobs/([\w-]+)/prs", path)
            if m:
                job, _ = self._job_bug(m.group(1))
                if job is None:
                    return self._err(404, "no such job")
                if not job.repo.get("can_open_prs"):
                    return self._err(400, "this repo is not on GitHub")
                todo = [b for b in job.bugs if b.status == "verified" and not (b.pr and b.pr.get("state") in ("open", "opening"))]
                for b in todo:
                    runner.open_pr(job, b)
                return self._json({"ok": True, "opening": [b.id for b in todo]}, 202)
            return self._err(404, "not found")

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, jobs_dir: str = "runs/web", backend: str = "docker",
          runner: JobRunner | None = None) -> ThreadingHTTPServer:
    runner = runner or JobRunner(JobStore(Path(jobs_dir)), backend=backend)
    httpd = ThreadingHTTPServer((host, port), make_handler(runner))
    httpd.daemon_threads = True
    return httpd


def cmd_serve(a) -> int:
    from ..sandbox import BUILD_HINT, _docker_available, _image_available, default_image

    backend = a.backend
    if backend == "docker" and not (_docker_available() and _image_available(default_image())):
        print(f"error: the web UI runs untrusted repo code and needs the docker sandbox.\n"
              f"Start Docker and build the image:\n  {BUILD_HINT}\n"
              f"(or pass --backend local to run on this machine without isolation, for trusted repos only)",
              file=sys.stderr)
        return 2
    if backend == "local":
        print("WARNING: --backend local runs repo code and the agent's commands directly on this machine.",
              file=sys.stderr)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: listening on {a.host}; anyone who can reach it can push with your GitHub credentials.",
              file=sys.stderr)
    httpd = serve(a.host, a.port, a.jobs_dir, backend)
    print(f"repo-fix UI on http://{a.host}:{httpd.server_address[1]}  (backend: {backend}, "
          f"GitHub auth: {'yes' if github_token() else 'no - PRs disabled until GITHUB_TOKEN or gh auth login'})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def register(subparsers) -> None:
    s = subparsers.add_parser("serve", help="web UI: repo URL in, verified fixes and PRs out")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--jobs-dir", default="runs/web")
    s.add_argument("--backend", choices=["docker", "local"], default="docker")
    s.set_defaults(fn=cmd_serve)
