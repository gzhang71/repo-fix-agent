"""Chat-template rendering with per-token loss weights (shared by the pipeline and scripts/train_lora.py).

Weights come from character spans of the rendered text: each trained assistant message gets its
weight; inside it, the reasoning text (`reasoning_content`, or reasoning placed in `content`) can
carry a different weight via the message's `reasoning_weight` key. Everything else (system
prompt, issue, tool outputs, the assistant header) has weight 0.
"""
from __future__ import annotations

TRAINING_KEYS = ("weight", "trainable", "reasoning_weight", "reasoning_text")


def clean(m: dict) -> dict:
    """Drop training-only keys before the chat template sees the message."""
    return {k: v for k, v in m.items() if k not in TRAINING_KEYS}


def render(tok, messages: list[dict], tools: list[dict] | None, gen_prompt: bool = False) -> str:
    return tok.apply_chat_template([clean(m) for m in messages], tools=tools or None, tokenize=False,
                                   add_generation_prompt=gen_prompt)


def weighted_spans(tok, messages: list[dict], weights: list[float], tools: list[dict] | None):
    """-> (text, spans) with spans [(start_char, end_char, weight)], most specific first."""
    text = render(tok, messages, tools)
    spans: list[tuple[int, int, float]] = []
    for i, (m, w) in enumerate(zip(messages, weights)):
        if w == 0 or m.get("role") != "assistant":
            continue
        start = len(render(tok, messages[:i], tools, gen_prompt=True)) if i else 0
        end = len(render(tok, messages[: i + 1], tools))
        rw = m.get("reasoning_weight")
        reasoning = m.get("reasoning_content") or m.get("reasoning_text") or ""
        if rw is not None and reasoning:
            at = text.find(reasoning, start, end)
            if at >= 0:   # the template may drop reasoning from earlier turns; then nothing to weight
                spans.append((at, at + len(reasoning), w * float(rw)))
        spans.append((start, end, w))
    return text, spans


def token_weights(tok, text: str, spans, max_len: int | None = None):
    """-> (input_ids, per-token weights, offsets)."""
    kw = {"truncation": True, "max_length": max_len} if max_len else {}
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, **kw)
    tw = []
    for a, b in enc["offset_mapping"]:
        tw.append(next((w for s, e, w in spans if s <= a < e), 0.0) if b > a else 0.0)
    return enc["input_ids"], tw, [tuple(o) for o in enc["offset_mapping"]]
