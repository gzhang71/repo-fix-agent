"""Single-turn text LLM interface for pipeline steps that need a model but not the agent loop
(thought backfill, issue writing/rephrasing, bug-type labeling, judging).

Pipeline code takes any `TextLLM` (a callable `(system, prompt) -> str`), so tests pass a plain
function and production passes `make_text_llm("claude-opus-5-5")` or `make_text_llm("gpt-5")`.
"""
from __future__ import annotations

from typing import Protocol

from .llm_cache import cached_call


class TextLLM(Protocol):
    def __call__(self, system: str, prompt: str) -> str: ...


class LLMRefusal(RuntimeError):
    pass


class AnthropicTextLLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def __call__(self, system: str, prompt: str) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                      messages=[{"role": "user", "content": prompt}])
        resp = cached_call("anthropic:messages", kwargs, lambda: self.client.messages.create(**kwargs))
        if resp.stop_reason == "refusal":
            raise LLMRefusal(f"{self.model} refused")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAITextLLM:
    def __init__(self, model: str = "gpt-5", reasoning_effort: str = "medium", max_completion_tokens: int = 16000):
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.reasoning_effort, self.max_completion_tokens = model, reasoning_effort, max_completion_tokens

    def __call__(self, system: str, prompt: str) -> str:
        kwargs = dict(model=self.model, reasoning_effort=self.reasoning_effort,
                      max_completion_tokens=self.max_completion_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        resp = cached_call(f"chat:{getattr(self.client, 'base_url', '')}", kwargs,
                           lambda: self.client.chat.completions.create(**kwargs))
        return (resp.choices[0].message.content or "").strip()


def make_text_llm(model: str) -> TextLLM:
    if model.startswith("claude-"):
        return AnthropicTextLLM(model)
    if model.startswith("gpt-") or (model[:1] == "o" and model[1:2].isdigit()):
        return OpenAITextLLM(model)
    raise ValueError(f"No text LLM backend for {model!r}")
