#!/usr/bin/env python3
"""LoRA training for the student: SFT, policy gradient (GRPO advantages), or DPO.

Runs on any machine with a GPU (transformers + peft; no TRL dependency):

  pip install torch transformers peft
  # SFT on chat fine-tuning JSONL (repo-fix export-chat-ft): per-message `weight` is the loss weight
  python scripts/train_lora.py --base <hf-id> --data runs/train/r1/chat_ft.jsonl --objective sft --out lora_sft
  # DPO on preference pairs (repo-fix prefs / onpolicy-distill --pairs-out), from the SFT adapter
  python scripts/train_lora.py --base <hf-id> --adapter lora_sft --objective dpo \
      --data runs/train/r1/prefs.jsonl --data runs/train/r1/onpolicy_pairs.jsonl --out lora_dpo
  # one GRPO-style update on an rl-rollouts batch (advantage-weighted log-likelihood + KL to reference)
  python scripts/train_lora.py --base <hf-id> --adapter lora_dpo --objective pg \
      --data runs/train/r1/rl_batch.jsonl --out lora_rl

Loss masking: each assistant turn's tokens get its weight (SFT: the exported `weight`; PG: the
episode's advantage on non-prefix steps; DPO: 1 on the chosen/rejected turns); everything else
(system prompt, issue, tool outputs, the assistant header) has weight 0. The reference model for
the KL / DPO terms is the base with the LoRA adapter switched off, so no second copy is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

DEFAULT_TEMPLATE_NOTE = "the tokenizer must have a chat template that renders `tools` and `tool_calls`"


# ---- data -----------------------------------------------------------------------------------------
def load_jsonl(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        out += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return out


def default_tools() -> list[dict]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from repo_fix_agent.sft import tools_as_functions
        return tools_as_functions()
    except Exception:  # noqa: BLE001
        return []


def _clean(m: dict) -> dict:
    """Drop training-only keys before the chat template sees the message."""
    return {k: v for k, v in m.items() if k not in ("weight", "trainable")}


def message_weights(messages: list[dict], objective: str, record: dict) -> list[float]:
    if objective == "sft":
        return [float(m.get("weight", 1.0)) if m.get("role") == "assistant" else 0.0 for m in messages]
    if objective == "pg":   # rl.episode_to_record: loss_mask marks the policy's own steps (not the prefix)
        adv = float(record.get("advantage_grpo", 0.0))
        mask = record.get("loss_mask") or [m.get("role") == "assistant" for m in messages]
        return [adv if (train and m.get("role") == "assistant") else 0.0 for m, train in zip(messages, mask)]
    raise ValueError(objective)


class Encoder:
    """Chat-template rendering with per-token weights, via character spans of each message."""

    def __init__(self, tok, max_len: int):
        self.tok, self.max_len = tok, max_len

    def render(self, messages: list[dict], tools: list[dict], gen_prompt: bool = False) -> str:
        return self.tok.apply_chat_template([_clean(m) for m in messages], tools=tools or None, tokenize=False,
                                            add_generation_prompt=gen_prompt)

    def encode(self, messages: list[dict], weights: list[float], tools: list[dict]):
        text = self.render(messages, tools)
        spans = []   # (start_char, end_char, weight) of each weighted assistant turn, excluding its header
        for i, (m, w) in enumerate(zip(messages, weights)):
            if w == 0 or m.get("role") != "assistant":
                continue
            start = len(self.render(messages[:i], tools, gen_prompt=True)) if i else 0
            end = len(self.render(messages[: i + 1], tools))
            spans.append((start, end, w))
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True,
                       max_length=self.max_len)
        tw = []
        for a, b in enc["offset_mapping"]:
            tw.append(next((w for s, e, w in spans if s <= a < e), 0.0) if b > a else 0.0)
        return enc["input_ids"], tw


# ---- losses ---------------------------------------------------------------------------------------
def token_logps(model, ids, torch):
    x = torch.tensor([ids], device=model.device)
    logits = model(input_ids=x).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)   # [T-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF model id or path of the student base weights")
    ap.add_argument("--adapter", default=None, help="start from an existing LoRA adapter")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--objective", choices=["sft", "pg", "dpo"], default="sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--kl-beta", type=float, default=0.05, help="pg: KL(policy||reference) coefficient")
    ap.add_argument("--adv-clip", type=float, default=5.0, help="pg: clip |advantage|")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--chat-template-file", default=None, help="override the tokenizer's chat template (Jinja)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=None)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.chat_template_file:
        tok.chat_template = Path(a.chat_template_file).read_text()
    if not tok.chat_template:
        raise SystemExit(f"no chat template: {DEFAULT_TEMPLATE_NOTE} (pass --chat-template-file)")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    enc = Encoder(tok, a.max_len)
    tools_default = default_tools()

    records = load_jsonl(a.data)
    if a.max_examples:
        records = records[: a.max_examples]
    if not records:
        raise SystemExit("no training records")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(records) * a.epochs)
    order = [i % len(records) for i in range(steps)]
    random.shuffle(order)
    model.train()
    running, skipped = 0.0, 0
    for step, idx in enumerate(order, 1):
        r = records[idx]
        tools = r.get("tools") or tools_default
        if a.objective == "dpo":
            loss = dpo_loss(model, enc, r, tools, a.dpo_beta, torch)
        else:
            msgs = r["messages"]
            w = message_weights(msgs, a.objective, r)
            if a.objective == "pg":
                w = [max(-a.adv_clip, min(a.adv_clip, x)) for x in w]
            ids, tw = enc.encode(msgs, w, tools)
            weights = torch.tensor(tw[1:], device=model.device)
            if not weights.abs().sum():
                skipped += 1
                continue
            logp = token_logps(model, ids, torch)
            mask = weights != 0
            if a.objective == "sft":
                loss = -(weights * logp).sum() / mask.sum()
            else:   # REINFORCE with group-relative advantages + per-token KL to the reference
                with torch.no_grad(), model.disable_adapter():
                    ref = token_logps(model, ids, torch)
                kl = (logp - ref)[mask].mean()
                loss = -(weights * logp)[mask].mean() + a.kl_beta * kl
        (loss / a.grad_accum).backward()
        running += float(loss)
        if step % a.grad_accum == 0 or step == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            print(f"step {step}/{len(order)}  loss {running / a.grad_accum:.4f}", flush=True)
            running = 0.0
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved adapter to {a.out} ({skipped} records had no trainable tokens)")
    return 0


def dpo_loss(model, enc, r: dict, tools, beta: float, torch):
    """Pair record: {prompt: [msgs], chosen: [assistant msg(s)], rejected: [...]}."""
    def seq_logp(completion, use_ref: bool):
        msgs = r["prompt"] + completion
        w = [0.0] * len(r["prompt"]) + [1.0 if m.get("role") == "assistant" else 0.0 for m in completion]
        ids, tw = enc.encode(msgs, w, tools)
        weights = torch.tensor(tw[1:], device=model.device)
        if use_ref:
            with torch.no_grad(), model.disable_adapter():
                return (token_logps(model, ids, torch) * weights).sum()
        return (token_logps(model, ids, torch) * weights).sum()

    pc, pr = seq_logp(r["chosen"], False), seq_logp(r["rejected"], False)
    rc, rr = seq_logp(r["chosen"], True), seq_logp(r["rejected"], True)
    return -torch.nn.functional.logsigmoid(beta * ((pc - rc) - (pr - rr)))


if __name__ == "__main__":
    sys.exit(main())
