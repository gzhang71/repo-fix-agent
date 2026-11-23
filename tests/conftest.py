import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_cache(monkeypatch, tmp_path):
    """Tests must never read or write the developer's LLM cache (.cache/llm). test_llm_cache.py
    re-enables it against a temp file."""
    monkeypatch.setenv("REPOFIX_LLM_CACHE", "off")
    monkeypatch.setenv("REPOFIX_LLM_CACHE_PATH", str(tmp_path / "llm_cache.sqlite"))
