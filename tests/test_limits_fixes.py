"""Secret scanner (patterns, entropy, PII false positives), parallel rollouts, sandbox selection."""
import threading
import time
from pathlib import Path

import pytest

from repo_fix_agent import sandbox as sb
from repo_fix_agent.grader import GradeResult
from repo_fix_agent.pipeline import audit as A
from repo_fix_agent.pipeline import distill as D
from repo_fix_agent.pipeline import rl
from repo_fix_agent.pipeline.parallel import pmap
from repo_fix_agent.policies import ScriptedPolicy
from repo_fix_agent.schema import Action, load_demos

ROOT = Path(__file__).resolve().parents[1]


# ---- secrets ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind", [
    ("ANTHROPIC_API_KEY=sk-ant-api03-" + "a1B2" * 10, "anthropic_key"),
    ("key = 'sk-proj-" + "Zx9" * 14 + "'", "openai_key"),
    ("AIza" + "SyD3x" * 7, "google_api_key"),
    ("stripe.api_key = 'sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc" + "'", "stripe_key"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "jwt"),
    ("DATABASE_URL=postgres://svc:hunter2pass@db.internal:5432/app", "url_credentials"),
    ('client = Client(key="Zx9qL2mN8vT4rY7wK3pB6sD1")', "high_entropy_string"),
    ('signing_secret = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"', "high_entropy_string"),
    ("mail jane.doe@acme-corp.com", "email"),
    ("connect to 10.24.3.17", "ip_address"),
])
def test_scanner_finds(text, kind):
    assert kind in A.scan_text(text)
    assert "[REDACTED:" in A.redact_text(text)


@pytest.mark.parametrize("text", [
    "commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b",                       # git sha, no secret context
    "index c633ca9..a71270b 100644",
    "PYTHONDONTWRITEBYTECODE test_parse_handles_iso_week_year_boundary",
    "src/repo_fix_agent/pipeline/augment.py",
    "id 123e4567-e89b-12d3-a456-426614174000",
    "https://user:password@example.com",
    "users: ann@x.io, a@b.io, bob@example.com, ci@build.test",                # fixture addresses
    "upgrade from 1.4.2.1 to 1.4.3",                                          # version, not an IP
])
def test_scanner_ignores(text):
    assert A.scan_text(text) == []


def test_seed_dataset_has_no_findings():
    assert {d.instance_id: A.scan_secrets(d) for d in load_demos(ROOT / "data/demos/seed.jsonl")
            if A.scan_secrets(d)} == {}


def test_detect_secrets_mode_requires_package(monkeypatch):
    try:
        import detect_secrets  # noqa: F401
        pytest.skip("detect-secrets installed")
    except ImportError:
        pass
    monkeypatch.setattr(A, "SECRET_SCANNER", "detect-secrets")
    with pytest.raises(RuntimeError, match="pip install detect-secrets"):
        A.scan_text("x")
    monkeypatch.setattr(A, "SECRET_SCANNER", "auto")
    assert A.scan_text("plain text") == []


# ---- parallel rollouts -----------------------------------------------------------------------
def test_pmap_preserves_order_and_runs_concurrently():
    active, peak, lock = [0], [0], threading.Lock()

    def f(x):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return x * x

    assert pmap(f, range(8), workers=4) == [x * x for x in range(8)]
    assert peak[0] > 1
    assert pmap(f, range(3), workers=1) == [0, 1, 4]


@pytest.fixture
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


def _fake_grade(demo, patch, root):
    return GradeResult(bool(patch.strip()))


def test_distill_parallel_matches_sequential(pag):
    def factory():
        return ScriptedPolicy([(s.thought, s.action) for s in pag.trajectory])

    kw = dict(k=3, hints=False, backend="local", grade_fn=_fake_grade, max_per_instance=3)
    seq = D.distill([pag], factory, ROOT, **kw)
    par = D.distill([pag], factory, ROOT, workers=3, **kw)
    strip = lambda rs: [(r.metadata["sample_idx"], r.patch, r.metadata["resolved"]) for r in rs]  # noqa: E731
    assert strip(seq[0]) == strip(par[0]) and seq[2]["cost"] == par[2]["cost"]


def test_rl_group_parallel(pag):
    good = [(s.thought, s.action) for s in pag.trajectory]
    eps = rl.rollout_group(lambda: ScriptedPolicy(good), pag, group_size=3, root=ROOT, backend="local",
                           grade_fn=_fake_grade, workers=3)
    assert [e.sample_idx for e in eps] == [0, 1, 2] and all(e.resolved for e in eps)
    give_up = rl.rollout_group(lambda: ScriptedPolicy([("x", Action("submit", {}))]), pag, group_size=2,
                               root=ROOT, backend="local", grade_fn=_fake_grade, workers=2)
    assert not any(e.resolved for e in give_up)


# ---- sandbox selection -----------------------------------------------------------------------
def test_docker_run_args_user():
    args = sb._docker_run_args("/r", "img", sb.SandboxLimits(), user="501:20")
    assert args[args.index("--user") + 1] == "501:20"


def test_env_var_forces_backend(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.setenv("REPOFIX_SANDBOX", "local")
    monkeypatch.setattr(sb, "_docker_available", lambda: pytest.fail("must not probe docker"))
    with sb.make_sandbox(str(tmp_path)) as s:
        assert isinstance(s, sb.LocalSandbox)


def test_auto_falls_back_when_image_missing(monkeypatch, tmp_path, caplog):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.delenv("REPOFIX_SANDBOX", raising=False)
    monkeypatch.setattr(sb, "_docker_available", lambda: True)
    monkeypatch.setattr(sb, "_image_available", lambda image: False)
    monkeypatch.setattr(sb, "_warned_fallback", False)
    with sb.make_sandbox(str(tmp_path), image="nope:1") as s:
        assert isinstance(s, sb.LocalSandbox)
    assert any("nope:1" in r.message and "docker build" in r.message for r in caplog.records)


def test_default_image_env(monkeypatch):
    monkeypatch.delenv("REPOFIX_SANDBOX_IMAGE", raising=False)
    assert sb.default_image() == "repo-fix-sandbox:py3.12"
    monkeypatch.setenv("REPOFIX_SANDBOX_IMAGE", "custom:2")
    assert sb.default_image() == "custom:2"
