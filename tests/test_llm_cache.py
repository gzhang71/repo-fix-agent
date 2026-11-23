"""LLM request/response cache (record/replay) and the .env loader."""
import os
from types import SimpleNamespace

import pytest

from repo_fix_agent import envfile, llm_cache
from repo_fix_agent.policies import LocalModelPolicy, StudentServingConfig


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOFIX_LLM_CACHE_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.delenv("REPOFIX_LLM_CACHE", raising=False)
    c = llm_cache.get_cache()
    yield c
    llm_cache._cache = None


def fresh_process(c):
    c._occ.clear()   # occurrence counters are per process


def test_sampled_requests_record_then_replay(cache):
    calls = []

    def fn():
        calls.append(1)
        return {"n": len(calls)}

    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    first = [llm_cache.cached_call("p", req, fn) for _ in range(3)]
    assert first == [{"n": 1}, {"n": 2}, {"n": 3}] and len(calls) == 3   # k samples stay distinct
    fresh_process(cache)
    again = [llm_cache.cached_call("p", req, fn) for _ in range(3)]
    assert again == first and len(calls) == 3                            # replayed, no new calls
    assert llm_cache.cached_call("p", req, fn) == {"n": 4}               # a 4th sample is new


def test_temperature_zero_dedupes(cache):
    n = []
    req = {"model": "m", "temperature": 0, "messages": []}
    for _ in range(3):
        llm_cache.cached_call("p", req, lambda: n.append(1) or "x")
    assert len(n) == 1


def test_key_covers_provider_and_params(cache):
    n = []
    f = lambda: n.append(1) or "x"  # noqa: E731
    llm_cache.cached_call("a", {"model": "m", "temperature": 0}, f)
    llm_cache.cached_call("b", {"model": "m", "temperature": 0}, f)
    llm_cache.cached_call("a", {"model": "m2", "temperature": 0}, f)
    assert len(n) == 3


def test_errors_are_not_cached(cache):
    req = {"model": "m", "temperature": 0}
    with pytest.raises(RuntimeError):
        llm_cache.cached_call("p", req, lambda: (_ for _ in ()).throw(RuntimeError("429")))
    assert llm_cache.cached_call("p", req, lambda: "ok") == "ok"


def test_disabled(cache, monkeypatch):
    monkeypatch.setenv("REPOFIX_LLM_CACHE", "off")
    n = []
    for _ in range(2):
        llm_cache.cached_call("p", {"model": "m", "temperature": 0}, lambda: n.append(1))
    assert len(n) == 2


def test_sdk_objects_roundtrip_and_stats(cache):
    from anthropic.types.beta import BetaMessage
    msg = BetaMessage.construct(**{
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "look"}, {"type": "tool_use", "id": "tu1", "name": "search",
                                                      "input": {"pattern": "x"}}],
        "usage": {"input_tokens": 100, "output_tokens": 7, "brand_new_field": {"model": None}}})
    req = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    llm_cache.cached_call("anthropic:beta.messages", req, lambda: msg)
    fresh_process(cache)
    back = llm_cache.cached_call("anthropic:beta.messages", req, lambda: pytest.fail("should replay"))
    assert type(back).__name__ == "BetaMessage" and back.content[1].type == "tool_use"
    assert back.content[1].input == {"pattern": "x"} and back.usage.input_tokens == 100
    row = cache.stats()["rows"][0]
    assert (row["entries"], row["hits"], row["saved_input_tokens"]) == (1, 1, 100)
    stored = cache._db.execute("SELECT request FROM calls").fetchone()[0]
    assert "api_key" not in stored


class FakeChat:
    def __init__(self):
        self.n = 0
        self.base_url = "https://fake/v1"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        from openai.types.chat import ChatCompletion
        self.n += 1
        return ChatCompletion.model_validate({
            "id": f"c{self.n}", "object": "chat.completion", "created": 0, "model": kw["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command": "ls"}'}}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}})


def test_policy_calls_go_through_cache(cache):
    client = FakeChat()
    for _ in range(2):
        fresh_process(cache)
        pol = LocalModelPolicy(StudentServingConfig(model_name="small", temperature=0.7), client=client)
        pol.reset("sys", "task")
        assert pol.act(None)[1].tool == "bash"
    assert client.n == 1   # second run replayed


# ---- .env ----------------------------------------------------------------------------------------
def test_env_parse_and_load(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nA=1\nexport B='two words'\nC=\"q#x\"\nD=plain # trailing\nbad line\nE=\n")
    assert envfile.parse(f.read_text()) == {"A": "1", "B": "two words", "C": "q#x", "D": "plain", "E": ""}
    for k in "ABCDE":
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("A", "from-shell")
    loaded = envfile.load_env(f)
    assert os.environ["A"] == "from-shell" and "A" not in loaded   # real env wins
    assert os.environ["B"] == "two words" and set(loaded) == {"B", "C", "D", "E"}


def test_env_file_discovery(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("X=1\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.delenv("REPOFIX_ENV_FILE", raising=False)
    assert envfile.find_env_file(sub) == tmp_path / ".env"
