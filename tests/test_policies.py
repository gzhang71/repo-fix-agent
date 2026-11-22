"""LocalModelPolicy against a fake OpenAI-compatible server (no network, no openai package)."""
import json
from types import SimpleNamespace as NS

from repo_fix_agent.agent import run_agent
from repo_fix_agent.env import RepoEnv
from repo_fix_agent.policies import LocalModelPolicy, StudentServingConfig, make_policy
from repo_fix_agent.schema import load_demos


def _tool_msg(name, args, content="", call_id="c1"):
    tc = NS(id=call_id, function=NS(name=name, arguments=json.dumps(args) if not isinstance(args, str) else args))
    return NS(model="student", choices=[NS(message=NS(content=content, tool_calls=[tc]))])


class FakeClient:
    """Replays a script of responses and records every request."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.responses.pop(0)


def test_student_solves_seed_demo_end_to_end():
    demo = load_demos("data/demos/seed.jsonl")[0]  # pagination off-by-one
    script = [_tool_msg(s.action.tool, s.action.args, s.thought, f"c{i}") for i, s in enumerate(demo.trajectory)]
    client = FakeClient(script)
    with RepoEnv(demo.repo, backend="local") as env:
        res = run_agent(LocalModelPolicy(StudentServingConfig(), client=client), env,
                        demo.problem_statement, demo.test_cmd)
    assert res.submitted and res.patch == demo.patch
    req = client.requests[-1]
    assert req["model"] == "student" and req["parallel_tool_calls"] is False
    assert {t["function"]["name"] for t in req["tools"]} >= {"bash", "view_file", "str_replace", "submit"}
    assert [m["role"] for m in req["messages"][:4]] == ["system", "user", "assistant", "tool"]


def test_invalid_json_args_become_validation_error():
    client = FakeClient([_tool_msg("bash", "{not json")])
    p = LocalModelPolicy(StudentServingConfig(), client=client)
    p.reset("sys", "task")
    _, action = p.act(None)
    assert action.tool == "bash" and "_invalid_json" in action.args


def test_context_budget_elides_old_observations_first():
    cfg = StudentServingConfig(max_new_tokens=500, chars_per_token=1.0, keep_recent_observations=1)
    p = LocalModelPolicy(cfg, client=FakeClient([]))
    # room for tool schemas + prompt + ~2 of the 5 observations
    cfg.max_context_tokens = p._tokens([]) + cfg.max_new_tokens + 1600
    p.reset("sys", "task")
    for i in range(5):
        p.messages.append({"role": "assistant", "content": f"step {i}", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
        p.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"obs{i} " + "x" * 600})
    msgs = p._prepare_messages()
    tools = [m["content"] for m in msgs if m["role"] == "tool"]
    assert tools[0] == LocalModelPolicy.ELIDED                  # oldest dropped first
    assert tools[-1].startswith("obs4")                         # newest kept intact
    assert p._tokens(msgs) <= cfg.max_context_tokens - cfg.max_new_tokens
    assert all(m["content"] == f"step {i}" for i, m in enumerate(x for x in msgs if x["role"] == "assistant"))
    assert p.messages[2 + 1]["content"].startswith("obs0")      # original history untouched


def test_config_loads_serving_section():
    cfg = StudentServingConfig.load("configs/models.yaml")
    assert cfg.max_context_tokens == 65536 and cfg.base_url.startswith("http")


def test_routing():
    import pytest
    with pytest.raises(ValueError):
        make_policy("llama")
