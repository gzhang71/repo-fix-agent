"""Policies = anything that maps (task, observations so far) -> next (thought, action).

* ScriptedPolicy   - replays a fixed action list (used to record/replay engineer demos).
* AnthropicPolicy  - teacher candidate (Claude) via the Messages API with tool use.
* OpenAIPolicy     - teacher candidate (ChatGPT / GPT models).
* LocalModelPolicy - student served by vLLM / SGLang (OpenAI-compatible, native tool calls).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

from .llm_cache import cached_call
from .schema import Action, Step
from .tools import TOOLS


class Policy(Protocol):
    def reset(self, system: str, task: str) -> None: ...
    def act(self, last_observation: str | None) -> tuple[str, Action]: ...
    def preload(self, steps: list[Step]) -> None: ...  # optional: seed already-executed steps


class ScriptedPolicy:
    def __init__(self, steps: list[tuple[str, Action]]):
        self._steps = steps

    def reset(self, system: str, task: str) -> None:
        self._i = 0

    def preload(self, steps: list[Step]) -> None:
        """Prefix already executed by the caller; scripted policies just continue their script."""

    def act(self, last_observation: str | None) -> tuple[str, Action]:
        if self._i >= len(self._steps):
            return "Script exhausted.", Action("submit", {})
        step = self._steps[self._i]
        self._i += 1
        return step


class AnthropicPolicy:
    """Teacher policy. Keeps the native message list append-only so thinking blocks
    stay valid across turns; the agent loop only sees (thought, action) pairs."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 16000):
        import anthropic  # imported lazily so the student/eval path has no hard dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.tools = [{**t, "strict": True} for t in TOOLS]
        self.served_models: set[str] = set()  # record which model actually answered (fallbacks)

    def reset(self, system: str, task: str) -> None:
        self.system = system
        self.messages: list[dict] = [{"role": "user", "content": task}]
        self._pending_tool_id: str | None = None

    def preload(self, steps: list[Step]) -> None:
        """Seed history with already-executed steps (RL prefix-conditioned rollouts, §3g).
        Call right after reset(); the next act() continues from the last observation."""
        for i, st in enumerate(steps):
            if self._pending_tool_id is not None:
                self.messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": self._pending_tool_id, "content": prev_obs}]})
            tid = f"prefix_{i}"
            content = ([{"type": "text", "text": st.thought}] if st.thought else []) + [
                {"type": "tool_use", "id": tid, "name": st.action.tool, "input": st.action.args}]
            self.messages.append({"role": "assistant", "content": content})
            self._pending_tool_id, prev_obs = tid, st.observation

    def _call(self):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            # Forced tool_choice is rejected on this model family; steer via the prompt instead.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
            # Automatic prompt caching: the breakpoint follows the growing conversation, so each
            # step re-reads the previous steps from cache instead of paying full input price.
            cache_control={"type": "ephemeral"},
            # Server-side fallback on safety-classifier refusals. Remove if you need strict
            # single-model provenance for distillation data.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
        )
        resp = cached_call("anthropic:beta.messages", kwargs, lambda: self.client.beta.messages.create(**kwargs))
        self.served_models.add(resp.model)
        return resp

    def act(self, last_observation: str | None) -> tuple[str, Action]:
        if self._pending_tool_id is not None:
            self.messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": self._pending_tool_id,
                "content": last_observation or "",
            }]})
            self._pending_tool_id = None

        for _ in range(3):
            resp = self._call()
            if resp.stop_reason == "refusal":
                return "Model refused.", Action("submit", {})
            self.messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            # Kept apart for distillation: the reasoning channel (summarized thinking on this
            # model) and the visible text are trained differently (pipeline/teacher_signal.py).
            self.last_reasoning = "\n".join(getattr(b, "thinking", "") for b in resp.content
                                            if b.type == "thinking").strip()
            self.last_text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            thought = "\n".join(x for x in (self.last_reasoning, self.last_text) if x)
            if tool_uses:
                tu = tool_uses[0]
                self._pending_tool_id = tu.id
                return thought, Action(tu.name, dict(tu.input))
            self.messages.append({"role": "user", "content":
                "Continue by calling one of the tools. Call `submit` when the fix is verified."})
        return "No tool call after retries.", Action("submit", {})


class _ChatCompletionsPolicy:
    """Shared loop for any OpenAI-format Chat Completions endpoint (OpenAI itself, vLLM, SGLang).

    Messages and tools use the same shape `sft.render_messages` / `sft.tools_as_functions`
    produce, so what the student sees at inference matches what it was fine-tuned on.
    """

    def __init__(self, model: str, client, request_kwargs: dict | None = None, keep_reasoning: bool = False):
        from .sft import tools_as_functions

        self.client = client
        self.model = model
        self.request_kwargs = request_kwargs or {}
        self.tools = tools_as_functions()
        self.served_models: set[str] = set()
        # Reasoning models on vLLM / SGLang / Fireworks return their thinking in a separate
        # `reasoning_content` (or `reasoning`) field. Keeping it (as the step's thought, and sent
        # back in later turns) lets the model continue its plan instead of re-deriving it.
        self.keep_reasoning = keep_reasoning

    def reset(self, system: str, task: str) -> None:
        self.messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        self._pending_tool_id: str | None = None

    def _prepare_messages(self) -> list[dict]:
        return self.messages  # hook: LocalModelPolicy trims to the context budget

    def preload(self, steps: list[Step]) -> None:
        """Seed history with already-executed steps (RL prefix-conditioned rollouts, §3g).
        Same message shape as sft.render_messages. Call right after reset()."""
        for i, st in enumerate(steps):
            if self._pending_tool_id is not None:
                self.messages.append({"role": "tool", "tool_call_id": self._pending_tool_id, "content": prev_obs})
            tid = f"prefix_{i}"
            msg = {"role": "assistant", "content": "" if self.keep_reasoning else st.thought, "tool_calls": [{
                "id": tid, "type": "function",
                "function": {"name": st.action.tool, "arguments": json.dumps(st.action.args)}}]}
            if self.keep_reasoning and st.thought:
                msg["reasoning_content"] = st.thought
            self.messages.append(msg)
            self._pending_tool_id, prev_obs = tid, st.observation

    def act(self, last_observation: str | None) -> tuple[str, Action]:
        if self._pending_tool_id is not None:
            self.messages.append({"role": "tool", "tool_call_id": self._pending_tool_id,
                                  "content": last_observation or ""})
            self._pending_tool_id = None

        for _ in range(3):
            kwargs = dict(model=self.model, messages=self._prepare_messages(), tools=self.tools,
                          tool_choice="auto", parallel_tool_calls=False, **self.request_kwargs)
            resp = cached_call(f"chat:{getattr(self.client, 'base_url', '')}", kwargs,
                               lambda: self.client.chat.completions.create(**kwargs))
            self.served_models.add(resp.model)
            msg = resp.choices[0].message
            self.last_reasoning, self.last_text = reasoning_of(msg), (msg.content or "").strip()
            reasoning = self.last_reasoning if self.keep_reasoning else ""
            thought = "\n".join(x for x in (reasoning, (msg.content or "").strip()) if x)
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                out = {"role": "assistant", "content": msg.content or "", "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]}
                if reasoning:
                    out["reasoning_content"] = reasoning
                self.messages.append(out)
                self._pending_tool_id = tc.id
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    # Unknown key -> the tool layer returns a validation error the model can fix.
                    args = {"_invalid_json": tc.function.arguments}
                if not isinstance(args, dict):
                    args = {"_invalid_json": tc.function.arguments}
                return thought, Action(tc.function.name, args)
            out = {"role": "assistant", "content": msg.content or ""}
            if reasoning:
                out["reasoning_content"] = reasoning
            self.messages.append(out)
            self.messages.append({"role": "user", "content":
                "Continue by calling one of the tools. Call `submit` when the fix is verified."})
        return "No tool call after retries.", Action("submit", {})


def reasoning_of(msg) -> str:
    """Separate reasoning text from an OpenAI-format message (`reasoning_content` on vLLM /
    SGLang / Fireworks / DeepSeek, `reasoning` on some others); "" if absent."""
    for name in ("reasoning_content", "reasoning"):
        v = getattr(msg, name, None)
        if v is None and getattr(msg, "model_extra", None):
            v = msg.model_extra.get(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


class OpenAIPolicy(_ChatCompletionsPolicy):
    """Teacher candidate: OpenAI GPT models (ChatGPT family).

    Set the model id to the newest GPT model your account has (configs/models.yaml).
    Reasoning tokens aren't carried between Chat Completions turns. If multi-step quality falls
    short, port this class to the Responses API (previous_response_id), which keeps reasoning
    state across tool calls.
    """

    def __init__(self, model: str = "gpt-5", reasoning_effort: str = "high", max_completion_tokens: int = 16000):
        from openai import OpenAI  # optional dependency: pip install '.[openai]'

        super().__init__(model, OpenAI(), {"reasoning_effort": reasoning_effort,
                                           "max_completion_tokens": max_completion_tokens})


@dataclass
class StudentServingConfig:
    """`student.serving` in configs/models.yaml."""
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "student"              # the --served-model-name given to vLLM/SGLang
    api_key_env: str = "STUDENT_API_KEY"     # env var holding the key; "EMPTY" is used if unset
    max_context_tokens: int = 65536          # the model's context window
    max_new_tokens: int = 4096               # reserved for each response
    chars_per_token: float = 3.5             # rough estimate for budgeting; tune per tokenizer
    keep_recent_observations: int = 3        # never elide the N most recent tool outputs
    temperature: float = 0.0                 # 0 for eval; raise for RL / best-of-n sampling
    keep_reasoning: bool = True              # record + resend the model's reasoning_content
    spec_check: bool = True                  # behaviour/edge-case checklist in the system prompt
    submit_gate: bool = True                 # bounce one unverified submit per episode

    def agent_options(self):
        from .agent import AgentOptions
        return AgentOptions(spec_check=self.spec_check, submit_gate=self.submit_gate)

    @classmethod
    def load(cls, path: str | Path = "configs/models.yaml") -> "StudentServingConfig":
        import yaml

        raw = (yaml.safe_load(Path(path).read_text()) or {}).get("student", {}).get("serving", {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


class LocalModelPolicy(_ChatCompletionsPolicy):
    """Student policy on a self-hosted vLLM / SGLang OpenAI-compatible server with native tool calling.

    The server applies the model's chat template (tools included) and parses its tool calls, e.g.
        vllm serve <weights> --served-model-name student --max-model-len 65536 \
            --enable-auto-tool-choice --tool-call-parser <parser for the model family>

    Context budget: when the conversation exceeds max_context_tokens - max_new_tokens, the oldest
    tool observations are replaced with a placeholder (system prompt, issue, every assistant
    turn and the most recent observations are kept). If it still doesn't fit, the newest
    observation is cut down to keep its tail.
    """

    ELIDED = "[earlier tool output elided to fit the context window]"

    def __init__(self, cfg: StudentServingConfig | None = None, client=None):
        self.cfg = cfg or StudentServingConfig.load()
        if client is None:
            from openai import OpenAI  # client for the OpenAI-compatible protocol vLLM/SGLang serve

            client = OpenAI(base_url=self.cfg.base_url, api_key=os.environ.get(self.cfg.api_key_env, "EMPTY"))
        super().__init__(self.cfg.model_name, client,
                         {"max_tokens": self.cfg.max_new_tokens, "temperature": self.cfg.temperature},
                         keep_reasoning=self.cfg.keep_reasoning)
        self.agent_options = self.cfg.agent_options()

    def _tokens(self, msgs: list[dict]) -> int:
        return estimate_tokens(msgs, self.tools, self.cfg.chars_per_token)

    def _prepare_messages(self) -> list[dict]:
        return apply_context_budget(self.messages, self.tools, self.cfg)


def estimate_tokens(msgs: list[dict], tools: list[dict], chars_per_token: float) -> int:
    chars = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in msgs)
    return int((chars + len(json.dumps(tools))) / chars_per_token)


def apply_context_budget(messages: list[dict], tools: list[dict], cfg: StudentServingConfig) -> list[dict]:
    """The ONE truncation policy shared by inference (LocalModelPolicy) and SFT data building, so
    training contexts match deployment exactly. Returns a trimmed copy; input is not modified."""
    budget = cfg.max_context_tokens - cfg.max_new_tokens
    msgs = [dict(m) for m in messages]
    tok = lambda: estimate_tokens(msgs, tools, cfg.chars_per_token)  # noqa: E731
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    elidable = tool_idx[:-cfg.keep_recent_observations] if cfg.keep_recent_observations else tool_idx
    for i in elidable:
        if tok() <= budget:
            return msgs
        msgs[i]["content"] = LocalModelPolicy.ELIDED
    over = tok() - budget
    if over > 0 and tool_idx:
        last = msgs[tool_idx[-1]]
        keep = max(200, len(last["content"]) - int(over * cfg.chars_per_token) - 100)
        last["content"] = "...[truncated to fit context]...\n" + last["content"][-keep:]
    return msgs


def make_policy(model: str, temperature: float | None = None) -> Policy:
    """Route a model id to its backend.

    claude-*        -> AnthropicPolicy (teacher)
    gpt-* / o<N>*   -> OpenAIPolicy (teacher)
    student         -> LocalModelPolicy using student.serving in configs/models.yaml
    student:<name>  -> same, overriding model_name
    `temperature` overrides the student's sampling temperature (teachers sample at their default).
    """
    if model.startswith("claude-"):
        return AnthropicPolicy(model=model)
    if model.startswith("gpt-") or (model[:1] == "o" and model[1:2].isdigit()):
        return OpenAIPolicy(model=model)
    if model == "student" or model.startswith("student:"):
        cfg = StudentServingConfig.load()
        if ":" in model:
            cfg.model_name = model.split(":", 1)[1]
        if temperature is not None:
            cfg.temperature = temperature   # sampling for pass@k / best-of-n / RL
        return LocalModelPolicy(cfg)
    raise ValueError(f"No backend for {model!r}. Use claude-*, gpt-*, or student[:<served-model-name>].")
