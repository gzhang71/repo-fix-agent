"""Pick the distillation method from what the teacher and student expose.

What a model exposes decides how much of the teacher's knowledge reaches the student:

  mode       teacher exposes                      student needs        training (scripts/train_lora.py)
  action     only its output (text + tool call)   -                    SFT on the teacher's action at states the
                                                                       student reached (DAgger); DPO at disagreements
  reasoning  its reasoning text (full or          -                    the same, plus the reasoning as a trained
             summarized)                                               target: in `reasoning_content` if the student
                                                                       has a reasoning channel, else in `content`.
                                                                       Summarized reasoning is a paraphrase, so it
                                                                       gets a lower weight (0.5) than raw reasoning
  topk       per-token log-probs (top-k) through  the SAME tokenizer   token-level on-policy distillation: the teacher
             an API (e.g. OpenAI-compatible       (HF id)              scores the student's own sampled tokens;
             /completions with echo + logprobs)                        loss = per-token reverse KL (policy gradient
                                                                       with reward log p_T - log p_S) + forward KL
                                                                       on the teacher's top-k (`--objective tokenkl`)
  logits     full logits (local weights)          the SAME tokenizer   GKD: full-distribution KL / JSD between student
                                                                       and teacher on the student's samples
                                                                       (`--objective gkd --teacher-base ...`)

Token-level modes need the same tokenizer because the loss compares probabilities of the same
token ids. Claude exposes summarized thinking and no log-probs, so with Claude the method is
`reasoning`; with a same-family open teacher on Fireworks it can be `topk`; with the teacher's
weights on your GPU box, `logits`.

  repo-fix distill-signal                         # what each model exposes -> chosen mode and why
  repo-fix token-distill --rollouts R --out O     # topk: teacher scores student episodes (dry run unless --yes)
  repo-fix token-distill --rollouts R --render-only --out O   # logits: student episodes as chat records (free)
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

REASONING = ("none", "summarized", "full")
LOGPROBS = ("none", "topk", "full")
MODES = ("action", "reasoning", "topk", "logits")


@dataclass
class ModelCaps:
    name: str
    reasoning: str = "none"           # does it expose its reasoning text? none | summarized | full
    logprobs: str = "none"            # none | topk (API) | full (local weights)
    tokenizer: str | None = None      # HF tokenizer id; token-level modes need teacher == student
    top_k: int = 5                    # topk: alternatives returned per position
    base_url: str | None = None       # topk: OpenAI-compatible endpoint that returns prompt log-probs
    api_key_env: str | None = None
    weights: str | None = None        # logits: HF weights path/id of the teacher

    def __post_init__(self):
        if self.reasoning not in REASONING:
            raise ValueError(f"{self.name}: reasoning must be one of {REASONING}, not {self.reasoning!r}")
        if self.logprobs not in LOGPROBS:
            raise ValueError(f"{self.name}: logprobs must be one of {LOGPROBS}, not {self.logprobs!r}")


def default_caps(name: str) -> ModelCaps:
    """What the known APIs expose today."""
    if name.startswith("claude-"):     # adaptive thinking returns summarized thinking; no log-probs
        return ModelCaps(name, reasoning="summarized")
    if name.startswith("gpt-") or (name[:1] == "o" and name[1:2].isdigit()):
        return ModelCaps(name)          # reasoning hidden on chat completions; no log-probs on reasoning models
    return ModelCaps(name)


def load_caps(path: str | Path = "configs/models.yaml") -> tuple[ModelCaps, ModelCaps]:
    """(teacher, student) from the `distillation_signal` section, filled from defaults."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
    sig = (cfg or {}).get("distillation_signal") or {}
    t = dict(sig.get("teacher") or {})
    tname = t.pop("name", None) or (((cfg or {}).get("teacher") or {}).get("primary") or {}).get("name", "claude-opus-5-5")
    teacher = ModelCaps(**{**asdict(default_caps(tname)), **t, "name": tname})
    serving = ((cfg or {}).get("student") or {}).get("serving") or {}
    s = dict(sig.get("student") or {})
    sname = s.pop("name", None) or serving.get("model_name", "student")
    s.setdefault("reasoning", "full" if serving.get("keep_reasoning") else "none")
    student = ModelCaps(**{**asdict(ModelCaps(sname)), **s, "name": sname})
    return teacher, student


@dataclass
class SignalPlan:
    mode: str
    why: list[str] = field(default_factory=list)
    reasoning_target: str | None = None    # reasoning modes: "reasoning_content" | "content"
    reasoning_weight: float = 0.0          # loss weight of the teacher's reasoning relative to its action
    train_objectives: list[str] = field(default_factory=list)


def same_tokenizer(t: ModelCaps, s: ModelCaps) -> bool:
    return bool(t.tokenizer and s.tokenizer and t.tokenizer == s.tokenizer)


def select_mode(teacher: ModelCaps, student: ModelCaps, prefer: str | None = None) -> SignalPlan:
    """Richest signal the pair supports (logits > topk > reasoning > action), or `prefer` if possible."""
    why, avail = [], ["action"]
    if teacher.reasoning != "none":
        avail.append("reasoning")
    else:
        why.append(f"{teacher.name} exposes no reasoning text")
    if teacher.logprobs == "none":
        why.append(f"{teacher.name} exposes no log-probs, so no token-level distillation")
    elif not same_tokenizer(teacher, student):
        why.append(f"{teacher.name} has log-probs but a different (or unknown) tokenizer from the student "
                   f"({teacher.tokenizer} vs {student.tokenizer}); token-level losses need the same token ids")
    else:
        if teacher.base_url:            # an API that returns prompt log-probs (full logits can serve top-k too)
            avail.append("topk")
        else:
            why.append("topk needs teacher.base_url (an endpoint that returns prompt log-probs)")
        if teacher.logprobs == "full":
            if teacher.weights:
                avail.append("logits")
            else:
                why.append("logits needs teacher.weights (HF weights loaded on the training box)")
    if prefer:
        if prefer not in avail:
            raise ValueError(f"mode {prefer!r} not supported by this pair: {'; '.join(why) or 'n/a'}")
        mode = prefer
    else:
        mode = avail[-1]
    plan = SignalPlan(mode, why)
    if teacher.reasoning != "none":   # reasoning text is useful context/target in every mode that has it
        plan.reasoning_target = "reasoning_content" if student.reasoning != "none" else "content"
        plan.reasoning_weight = 1.0 if teacher.reasoning == "full" else 0.5
    plan.train_objectives = {
        "action": ["sft (on-policy relabels)", "dpo (disagreement pairs)"],
        "reasoning": ["sft (relabels with reasoning, reasoning-weighted)", "dpo (disagreement pairs)"],
        "topk": ["sft warm start (optional)", "tokenkl (teacher log-probs on student tokens)"],
        "logits": ["sft warm start (optional)", "gkd (full-distribution KL to the local teacher)"],
    }[mode]
    return plan


# ---- reasoning / action records ---------------------------------------------------------------------
def shape_target(reasoning: str, text: str, plan: SignalPlan | None) -> dict:
    """The trained assistant message for one teacher step (tool_calls added by the caller).
    action mode: visible text only. reasoning mode: the reasoning goes where the student keeps its
    own (reasoning_content) or in front of the text, with its own loss weight."""
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if not plan or plan.mode == "action" or not reasoning or plan.reasoning_weight <= 0:
        return msg
    if plan.reasoning_target == "reasoning_content":
        msg["reasoning_content"] = reasoning
    else:
        msg["content"] = f"{reasoning}\n\n{text}".strip()
        msg["reasoning_text"] = reasoning          # lets train_lora find the span inside content
    msg["reasoning_weight"] = plan.reasoning_weight
    return msg


# ---- token-level (topk): teacher scores the student's own tokens -----------------------------------
class TokenizerMismatch(ValueError):
    """Teacher tokens don't line up with the student's: token-level distillation is invalid."""


class CompletionsScorer:
    """Prompt log-probs from an OpenAI-compatible /completions endpoint (echo=True, logprobs=k).
    Every request goes through the LLM cache, so re-scoring the same text costs nothing."""

    def __init__(self, caps: ModelCaps, client=None):
        from ..llm_cache import cached_call

        self.caps, self._cached = caps, cached_call
        if client is None:
            from openai import OpenAI

            key = os.environ.get(caps.api_key_env or "", "") or "EMPTY"
            client = OpenAI(base_url=caps.base_url, api_key=key)
        self.client = client

    def score(self, text: str) -> list[dict]:
        req = {"model": self.caps.name, "prompt": text, "max_tokens": 1, "echo": True,
               "logprobs": self.caps.top_k, "temperature": 0}
        resp = self._cached(f"completions:{self.caps.base_url}", req, lambda: self.client.completions.create(**req))
        lp = resp.choices[0].logprobs
        out = []
        for tok, off, l, top in zip(lp.tokens, lp.text_offset, lp.token_logprobs, lp.top_logprobs or [None] * len(lp.tokens)):
            if off >= len(text):        # the one generated token
                break
            out.append({"token": tok, "offset": off, "logprob": l, "top": dict(top or {})})
        return out


def align(offsets: list[tuple[int, int]], text: str, teacher: list[dict], mask: list[float],
          max_mismatch: float = 0.01) -> tuple[list[float | None], list[dict | None]]:
    """Per student token: the teacher's log-prob of that same token and its top-k, matched by
    character offset and token text. Raises if trained tokens disagree (different tokenizers)."""
    by_off = {t["offset"]: t for t in teacher}
    lps: list[float | None] = []
    tops: list[dict | None] = []
    bad = trained = 0
    for (a, b), w in zip(offsets, mask):
        t = by_off.get(a)
        ok = t is not None and t["token"] == text[a:b]
        if w:
            trained += 1
            bad += not ok
        lps.append(t["logprob"] if ok else None)
        tops.append(t["top"] if ok and t["top"] else None)
    if trained and bad / trained > max_mismatch:
        raise TokenizerMismatch(f"{bad}/{trained} trained tokens don't match the teacher's tokenization; "
                                "token-level distillation needs the same tokenizer")
    return lps, tops


def topk_ids(tok, tops: list[dict | None]) -> list[list[list[float]] | None]:
    """Teacher alternatives (token text -> log-prob) as [[student token id, log-prob], ...];
    alternatives that aren't a single token for this tokenizer are dropped."""
    out = []
    for top in tops:
        if not top:
            out.append(None)
            continue
        pairs = []
        for s, l in top.items():
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and l is not None:
                pairs.append([ids[0], float(l)])
        out.append(pairs or None)
    return out


def token_record(rec: dict, tok, scorer: CompletionsScorer, max_len: int | None = None,
                 max_mismatch: float = 0.01) -> dict:
    """Chat record (messages, tools, loss_mask) of a student episode -> token-level training record."""
    from ..chat_spans import token_weights, weighted_spans

    mask = rec.get("loss_mask") or [m.get("role") == "assistant" for m in rec["messages"]]
    text, spans = weighted_spans(tok, rec["messages"], [1.0 if x else 0.0 for x in mask], rec.get("tools"))
    ids, tw, offs = token_weights(tok, text, spans, max_len)
    teacher = scorer.score(text)
    lps, tops = align(offs, text, teacher, tw, max_mismatch)
    return {"id": rec.get("id"), "input_ids": ids, "mask": [1 if w else 0 for w in tw], "teacher_logprobs": lps,
            "teacher_topk": topk_ids(tok, tops), "meta": rec.get("meta") or {}}


def rollout_records(rollouts: list, cfg=None) -> list[dict]:
    """Student episodes as chat records under the student's prompt (all assistant turns trained)."""
    from .sft_data import instance_key, render_record

    out = []
    for d in rollouts:
        r = render_record(d, cfg, d.metadata.get("source") or "student", d.instance_id)
        r["meta"] = {"parent_id": instance_key(d), "resolved": d.metadata.get("resolved")}
        out.append(r)
    return out


def estimate_tokens(records: list[dict], chars_per_token: float = 3.5) -> int:
    return sum(math.ceil(sum(len(json.dumps(m)) for m in r["messages"]) / chars_per_token) for r in records)


# ---- CLI ------------------------------------------------------------------------------------------
def format_plan(teacher: ModelCaps, student: ModelCaps, plan: SignalPlan) -> str:
    lines = [f"teacher {teacher.name}: reasoning={teacher.reasoning}, logprobs={teacher.logprobs}, "
             f"tokenizer={teacher.tokenizer}",
             f"student {student.name}: reasoning={student.reasoning}, tokenizer={student.tokenizer}",
             f"\nmode: {plan.mode}"]
    if plan.reasoning_target:
        lines.append(f"  teacher reasoning -> student {plan.reasoning_target}, loss weight {plan.reasoning_weight}")
    lines += [f"  train: {o}" for o in plan.train_objectives]
    if plan.why:
        lines.append("why not a richer mode:")
        lines += [f"  - {w}" for w in plan.why]
    return "\n".join(lines)


def cmd_signal(a) -> int:
    teacher, student = load_caps(a.config)
    try:
        plan = select_mode(teacher, student, a.mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(plan)) if a.json else format_plan(teacher, student, plan))
    return 0


def cmd_token_distill(a) -> int:
    from ..policies import StudentServingConfig
    from ..schema import load_demos
    from .sft_data import check_excluded, load_exclude_ids, write_jsonl

    rollouts = [d for d in load_demos(a.rollouts) if d.trajectory]
    if a.only_resolved:
        rollouts = [d for d in rollouts if d.metadata.get("resolved")]
    check_excluded(rollouts, load_exclude_ids(a.exclude_ids), "student rollouts")
    cfg = StudentServingConfig.load(a.config) if Path(a.config).exists() else StudentServingConfig()
    records = rollout_records(rollouts, cfg)
    if a.render_only:   # logits mode: train_lora computes the teacher's distribution itself
        n = write_jsonl(records, a.out)
        print(f"wrote {n} chat records to {a.out} (for --objective gkd)")
        return 0
    teacher, student = load_caps(a.config)
    plan = select_mode(teacher, student)
    if plan.mode not in ("topk", "logits"):
        print(f"token-distill needs a same-tokenizer teacher with log-probs; this pair is mode {plan.mode!r}:",
              file=sys.stderr)
        for w in plan.why:
            print(f"  - {w}", file=sys.stderr)
        return 2
    toks = estimate_tokens(records)
    print(json.dumps({"plan": {"episodes": len(records), "teacher": teacher.name, "prompt_tokens_est": toks,
                               "cost_est_usd": round(toks * a.price_per_mtok / 1e6, 2)}}), file=sys.stderr)
    if not a.yes:
        print("dry run: no teacher calls made (re-run with --yes)")
        return 0
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(student.tokenizer)
    scorer = CompletionsScorer(teacher)
    out, skipped = [], 0
    for r in records:
        try:
            out.append(token_record(r, tok, scorer, a.max_len, a.max_mismatch))
        except TokenizerMismatch as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # noqa: BLE001 - one bad request must not lose the batch
            skipped += 1
            print(f"skip {r.get('id')}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    n = write_jsonl(out, a.out)
    print(f"wrote {n} token-level records to {a.out} ({skipped} skipped)")
    return 0


def register(subparsers) -> None:
    s = subparsers.add_parser("distill-signal", help="what the teacher/student expose -> distillation mode (free)")
    s.add_argument("--config", default="configs/models.yaml")
    s.add_argument("--mode", choices=MODES, default=None, help="force a mode (error if the pair can't support it)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_signal)

    t = subparsers.add_parser("token-distill",
                              help="token-level data: same-tokenizer teacher scores student episodes (dry run unless --yes)")
    t.add_argument("--rollouts", required=True, help="student episodes (eval --save-rollouts)")
    t.add_argument("--out", required=True)
    t.add_argument("--render-only", action="store_true", help="just write chat records (logits / gkd mode; free)")
    t.add_argument("--only-resolved", action="store_true", help="score only episodes the student solved")
    t.add_argument("--max-len", type=int, default=32768)
    t.add_argument("--max-mismatch", type=float, default=0.01, help="tolerated fraction of misaligned trained tokens")
    t.add_argument("--price-per-mtok", type=float, default=0.9, help="teacher prompt price for the estimate")
    t.add_argument("--exclude-ids", default=None)
    t.add_argument("--config", default="configs/models.yaml")
    t.add_argument("--yes", action="store_true", help="make the (paid) teacher scoring calls")
    t.set_defaults(fn=cmd_token_distill)
