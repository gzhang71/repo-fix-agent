"""Convert demos into chat-format SFT examples (HF `tool_calls` convention).

The output is consumable by TRL / Axolotl / LLaMA-Factory with the student's chat template.
Loss should be masked to assistant turns only; tool observations are context, not targets.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agent import SYSTEM_PROMPT, build_task
from .schema import Demo, iter_demos
from .tools import TOOLS


def render_messages(demo: Demo, system_prompt: str | None = None, reasoning_field: bool = False) -> list[dict]:
    """`system_prompt` defaults to SYSTEM_PROMPT; pass the student's (AgentOptions.system_prompt())
    so training matches inference. `reasoning_field` puts each thought in `reasoning_content`
    (how reasoning students return and receive it) instead of `content`."""
    msgs: list[dict] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": build_task(demo.problem_statement, demo.test_cmd)},
    ]
    for i, step in enumerate(demo.trajectory):
        call_id = f"call_{i}"
        msg = {
            "role": "assistant",
            "content": "" if reasoning_field else step.thought,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": step.action.tool, "arguments": json.dumps(step.action.args)}}],
        }
        if reasoning_field and step.thought:
            msg["reasoning_content"] = step.thought
        msgs.append(msg)
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": step.observation})
    return msgs


def tools_as_functions() -> list[dict]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOLS]


def export_sft(demos_path: str | Path, out_path: str | Path) -> int:
    n = 0
    with open(out_path, "w") as f:
        for demo in iter_demos(demos_path):
            f.write(json.dumps({"id": demo.instance_id, "messages": render_messages(demo),
                                "tools": tools_as_functions()}) + "\n")
            n += 1
    return n
