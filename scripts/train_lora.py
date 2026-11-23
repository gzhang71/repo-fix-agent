#!/usr/bin/env python3
"""LoRA training for the student: SFT, policy gradient (GRPO advantages), DPO, or token-level
distillation (tokenkl: teacher log-probs from an API; gkd: full logits of a local teacher).

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
  # token-level on-policy distillation, teacher log-probs precomputed (repo-fix token-distill)
  python scripts/train_lora.py --base <hf-id> --adapter lora_sft --objective tokenkl \
      --data runs/train/r1/token_distill.jsonl --out lora_tok
  # GKD with the teacher's weights on this box (same tokenizer; repo-fix token-distill --render-only)
  python scripts/train_lora.py --base <hf-id> --adapter lora_sft --objective gkd --teacher-base <teacher-hf-id> \
      --data runs/train/r1/rollout_records.jsonl --out lora_gkd

Which of these applies depends on what the teacher exposes (`repo-fix distill-signal`).

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


def _spans_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from repo_fix_agent import chat_spans
    return chat_spans


def message_weights(messages: list[dict], objective: str, record: dict) -> list[float]:
    if objective in ("sft", "gkd"):
        return [float(m.get("weight", 1.0)) if m.get("role") == "assistant" else 0.0 for m in messages]
    if objective == "pg":   # rl.episode_to_record: loss_mask marks the policy's own steps (not the prefix)
        adv = float(record.get("advantage_grpo", 0.0))
        mask = record.get("loss_mask") or [m.get("role") == "assistant" for m in messages]
        return [adv if (train and m.get("role") == "assistant") else 0.0 for m, train in zip(messages, mask)]
    raise ValueError(objective)


class Encoder:
    """Chat-template rendering with per-token weights, via character spans of each message
    (repo_fix_agent.chat_spans; a message's `reasoning_weight` scales its reasoning tokens)."""

    def __init__(self, tok, max_len: int):
        self.tok, self.max_len, self.cs = tok, max_len, _spans_module()

    def render(self, messages: list[dict], tools: list[dict], gen_prompt: bool = False) -> str:
        return self.cs.render(self.tok, messages, tools, gen_prompt)

    def encode(self, messages: list[dict], weights: list[float], tools: list[dict]):
        text, spans = self.cs.weighted_spans(self.tok, messages, weights, tools)
        ids, tw, _ = self.cs.token_weights(self.tok, text, spans, self.max_len)
        return ids, tw


# ---- losses ---------------------------------------------------------------------------------------
def token_logps(model, ids, torch):
    x = torch.tensor([ids], device=model.device)
    logits = model(input_ids=x).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)   # [T-1]


def all_logps(model, ids, torch):
    x = torch.tensor([ids], device=model.device)
    return torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)                  # [T-1, V]


def tokenkl_loss(model, r: dict, max_len: int, adv_clip: float, topk_coef: float, torch):
    """On-policy distillation from teacher log-probs of the student's own tokens.

    Positions are next-token predictions: logits[j] predicts token j+1, so teacher_logprobs[j+1]
    (log p_T of token j+1) and teacher_topk[j+1] pair with logits[j].
      reverse KL: policy gradient with per-token reward log p_T(x) - log p_S(x) (no grad through
                  the reward); its expectation under the student's samples is -KL(p_S || p_T).
      top-k:      forward KL from the teacher's renormalized top-k onto the student's log-probs.
    """
    ids = r["input_ids"][:max_len]
    mask = torch.tensor([bool(m) for m in r["mask"][1:len(ids)]], device=model.device)
    t_lp = torch.tensor([float("nan") if v is None else v for v in r["teacher_logprobs"][1:len(ids)]],
                        device=model.device)
    mask = mask & ~torch.isnan(t_lp)
    if not mask.any():
        return None
    lp_all = all_logps(model, ids, torch)
    s_lp = lp_all.gather(-1, torch.tensor(ids[1:], device=model.device).unsqueeze(-1)).squeeze(-1)
    adv = (torch.nan_to_num(t_lp) - s_lp.detach()).clamp(-adv_clip, adv_clip)
    loss = -(adv * s_lp)[mask].mean()
    if topk_coef and r.get("teacher_topk"):
        terms = []
        for j, top in enumerate(r["teacher_topk"][1:len(ids)]):
            if not top or not bool(r["mask"][j + 1]):
                continue
            tid = torch.tensor([t for t, _ in top], device=model.device)
            p_t = torch.softmax(torch.tensor([l for _, l in top], device=model.device), -1)
            terms.append((p_t * (p_t.log() - lp_all[j, tid])).sum())
        if terms:
            loss = loss + topk_coef * torch.stack(terms).mean()
    return loss


def retain_kl(model, ids, weights, torch):
    """Anti-forgetting anchor: KL(p_adapter || p_base) over the trained positions of a retention
    record, exact over the vocabulary. The base is this same model with the LoRA adapter off, so
    no second copy is loaded. Zero when the adapter leaves these predictions unchanged."""
    mask = weights != 0
    if not mask.any():
        return None
    s = all_logps(model, ids, torch)[mask]
    with torch.no_grad(), model.disable_adapter():
        b = all_logps(model, ids, torch)[mask]
    return (s.exp() * (s - b)).sum(-1).mean()


def gkd_loss(model, teacher, ids, weights, beta: float, torch):
    """Generalized JSD between student and teacher next-token distributions at trained positions.
    beta=1: reverse KL(p_S || p_T) (mode-seeking; the usual on-policy choice); beta=0: forward
    KL(p_T || p_S); in between: beta*KL(p_T||m) + (1-beta)*KL(p_S||m), m = beta*p_T + (1-beta)*p_S."""
    mask = weights != 0
    s = all_logps(model, ids, torch)
    with torch.no_grad():
        t = all_logps(teacher, ids, torch).to(s.device)
    v = min(s.shape[-1], t.shape[-1])          # same tokenizer, but embedding tables may be padded differently
    s, t = s[mask, :v], t[mask, :v]
    if beta >= 1:
        kl = (s.exp() * (s - t)).sum(-1)
    elif beta <= 0:
        kl = (t.exp() * (t - s)).sum(-1)
    else:
        m = torch.logsumexp(torch.stack([t + math.log(beta), s + math.log(1 - beta)]), 0)
        kl = beta * (t.exp() * (t - m)).sum(-1) + (1 - beta) * (s.exp() * (s - m)).sum(-1)
    return (kl * weights[mask].abs()).sum() / mask.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF model id or path of the student base weights")
    ap.add_argument("--adapter", default=None, help="start from an existing LoRA adapter")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--objective", choices=["sft", "pg", "dpo", "tokenkl", "gkd"], default="sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--kl-beta", type=float, default=0.05, help="pg: KL(policy||reference) coefficient")
    ap.add_argument("--adv-clip", type=float, default=5.0, help="pg: clip |advantage|")
    ap.add_argument("--loss-agg", choices=["token", "sequence"], default="token",
                    help="pg: token = sum over every trained token in the accumulation window / their count "
                         "(long episodes aren't down-weighted); sequence = mean per episode, then over episodes")
    ap.add_argument("--clip-low", type=float, default=0.2, help="pg: ratio clip below (1 - clip_low); 0 = no clipping")
    ap.add_argument("--clip-high", type=float, default=0.28,
                    help="pg: ratio clip above (1 + clip_high); higher than clip-low so rare good tokens can grow")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--topk-coef", type=float, default=0.5, help="tokenkl: weight of the top-k forward-KL term")
    ap.add_argument("--teacher-base", default=None, help="gkd: HF id/path of the teacher (same tokenizer)")
    ap.add_argument("--gkd-beta", type=float, default=1.0, help="gkd: 1 = reverse KL, 0 = forward KL, else JSD")
    ap.add_argument("--retain-data", action="append", default=None,
                    help="retention records (repo-fix retain-set): penalise drift from the base on them (repeatable)")
    ap.add_argument("--retain-beta", type=float, default=0.1, help="weight of the retention KL term")
    ap.add_argument("--retain-every", type=int, default=4, help="one retention record every N training records")
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
    teacher = None
    if a.objective == "gkd":
        if not a.teacher_base:
            raise SystemExit("--objective gkd needs --teacher-base")
        t_tok = AutoTokenizer.from_pretrained(a.teacher_base)
        if t_tok.get_vocab() != tok.get_vocab():
            raise SystemExit("gkd needs the teacher and student to share a tokenizer (vocabularies differ)")
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher_base, torch_dtype=dtype,
                                                       device_map="auto" if torch.cuda.is_available() else None)
        teacher.eval().requires_grad_(False)
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
    pg_cache, old_lp, window_tokens = {}, {}, []
    if a.objective == "pg":
        # Encode once; record the starting policy's log-probs ("old" policy for the clipped ratio:
        # the adapter that generated the rollouts). With --epochs > 1 the ratio keeps later passes
        # from moving any token's probability too far from where sampling left it.
        for i, r in enumerate(records):
            w = [max(-a.adv_clip, min(a.adv_clip, x)) for x in message_weights(r["messages"], "pg", r)]
            ids, tw = enc.encode(r["messages"], w, r.get("tools") or tools_default)
            pg_cache[i] = (ids, tw)
            if a.clip_low > 0 and any(tw):
                with torch.no_grad():
                    old_lp[i] = token_logps(model, ids, torch).detach().to("cpu", torch.float32)
        for k in range(0, len(order), a.grad_accum):   # trained tokens per accumulation window
            window_tokens.append(max(1, sum(sum(1 for x in pg_cache[j][1][1:] if x) for j in order[k:k + a.grad_accum])))
    retain = load_jsonl(a.retain_data) if a.retain_data else []
    random.Random(a.seed + 1).shuffle(retain)
    retain_state = {"i": 0, "kl": 0.0, "n": 0}

    def retain_step() -> None:
        """Add one retention record's KL to the current accumulation window."""
        if not retain or a.retain_beta <= 0:
            return
        r = retain[retain_state["i"] % len(retain)]
        retain_state["i"] += 1
        mask = r.get("loss_mask") or [m.get("role") == "assistant" for m in r["messages"]]
        ids, tw = enc.encode(r["messages"], [1.0 if x else 0.0 for x in mask], r.get("tools") or tools_default)
        kl = retain_kl(model, ids, torch.tensor(tw[1:], device=model.device), torch)
        if kl is not None:
            (a.retain_beta * kl / a.grad_accum).backward()
            retain_state["kl"] += float(kl)
            retain_state["n"] += 1

    model.train()
    running, skipped = 0.0, 0

    def _maybe_step(step: int) -> None:
        nonlocal running
        if retain and step % max(1, a.retain_every) == 0:
            retain_step()
        if step % a.grad_accum == 0 or step == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            extra = ""
            if retain_state["n"]:
                extra = f"  retain_kl {retain_state['kl'] / retain_state['n']:.5f}"
                retain_state.update(kl=0.0, n=0)
            print(f"step {step}/{len(order)}  loss {running / a.grad_accum:.4f}{extra}", flush=True)
            running = 0.0
    for step, idx in enumerate(order, 1):
        r = records[idx]
        tools = r.get("tools") or tools_default
        if a.objective == "dpo":
            loss = dpo_loss(model, enc, r, tools, a.dpo_beta, torch)
        elif a.objective == "tokenkl":
            loss = tokenkl_loss(model, r, a.max_len, a.adv_clip, a.topk_coef, torch)
            if loss is None:
                skipped += 1
                _maybe_step(step)
                continue
        elif a.objective == "pg":
            ids, tw = pg_cache[idx]
            weights = torch.tensor(tw[1:], device=model.device)
            mask = weights != 0
            if not mask.any():
                skipped += 1
                _maybe_step(step)
                continue
            logp = token_logps(model, ids, torch)
            if idx in old_lp:   # PPO-style clipped surrogate, asymmetric ("clip-higher")
                ratio = torch.exp(logp - old_lp[idx].to(logp.device))
                surr = torch.min(ratio * weights, ratio.clamp(1 - a.clip_low, 1 + a.clip_high) * weights)
            else:               # plain REINFORCE with group-relative advantages
                surr = weights * logp
            with torch.no_grad(), model.disable_adapter():
                ref = token_logps(model, ids, torch)
            kl = logp - ref
            if a.loss_agg == "token":
                denom = window_tokens[(step - 1) // a.grad_accum]
                loss = (-surr[mask].sum() + a.kl_beta * kl[mask].sum()) / denom
                loss.backward()
                running += float(loss) * a.grad_accum
                _maybe_step(step)
                continue
            loss = -surr[mask].mean() + a.kl_beta * kl[mask].mean()
        else:
            msgs = r["messages"]
            w = message_weights(msgs, a.objective, r)
            ids, tw = enc.encode(msgs, w, tools)
            weights = torch.tensor(tw[1:], device=model.device)
            if not weights.abs().sum():
                skipped += 1
                _maybe_step(step)
                continue
            if a.objective == "gkd":
                loss = gkd_loss(model, teacher, ids, weights, a.gkd_beta, torch)
                (loss / a.grad_accum).backward()
                running += float(loss)
                _maybe_step(step)
                continue
            logp = token_logps(model, ids, torch)
            mask = weights != 0
            loss = -(weights * logp).sum() / mask.sum()   # sft
        (loss / a.grad_accum).backward()
        running += float(loss)
        _maybe_step(step)
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
