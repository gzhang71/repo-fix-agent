"""On-policy distillation with a black-box teacher (DAgger-style relabeling).

Off-policy distillation (`distill`) trains the student on the *teacher's* trajectories, so the
student never sees the states its own mistakes lead to. On-policy distillation trains on states
the *student* reaches:

  1. Roll out the current student (`repo-fix eval ... --save-rollouts`, temperature > 0).
  2. At (every k-th) state of each episode, show the teacher the task and the student's
     transcript so far and ask for the single best next tool call.
  3. Train the student to take the teacher's action from the student's own state
     (SFT record: student-format prefix, loss only on the teacher's turn). Repeat with the new
     student (DAgger rounds).

Token-level on-policy distillation (reverse KL against the teacher's per-token log-probs) needs
a teacher that exposes log-probs and shares the student's tokenizer; Claude doesn't expose them,
so this relabels actions instead. Disagreements also yield step-level DPO pairs (chosen =
teacher's action, rejected = student's action from the same state).

  repo-fix onpolicy-distill --rollouts runs/onpolicy/r1_student.jsonl --teacher claude-opus-5-5 \
      --out runs/onpolicy/r1_sft.jsonl --pairs-out runs/onpolicy/r1_pairs.jsonl --every 2 --dry-run
"""
from __future__ import annotations

import json
import math
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .. import sft
from ..agent import AgentOptions, build_task, student_options
from ..policies import StudentServingConfig, apply_context_budget
from ..schema import Action, Demo, Step, load_demos
from .parallel import pmap
from .verifier import ExcludedInstanceError, check_not_excluded, load_exclude_ids

RELABEL_INSTRUCTION = (
    "Above is the session so far: another engineer's steps (reasoning, tool call, output) on this "
    "issue, in order. Take over from the current state. Reply with the single best next tool call. "
    "If the work so far is wrong, correct course; if the fix is complete and verified, call `submit`.")


@dataclass
class Relabel:
    step: int                      # index of the state (number of student steps before it)
    student_thought: str
    student_action: Action | None  # None when the episode ended before this state
    teacher_thought: str
    teacher_action: Action
    agree: bool
    teacher_reasoning: str = ""    # the reasoning channel alone (thinking / reasoning_content), if exposed
    teacher_text: str | None = None   # the visible text alone; None = unknown (teacher_thought holds both)


# ---- prompting --------------------------------------------------------------------------------
def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n // 3] + "\n...[output truncated]...\n" + text[-(2 * n // 3):]


def transcript_blocks(steps: list[Step], max_obs_chars: int = 2500, keep_full_recent: int = 3) -> list[dict]:
    """One text block per step so prompt caching can reuse the shared prefix across relabels of
    the same episode. Old observations are truncated harder than recent ones."""
    blocks = []
    for i, st in enumerate(steps):
        recent = i >= len(steps) - keep_full_recent
        obs = _clip(st.observation, max_obs_chars if recent else max_obs_chars // 4)
        thought = f"Reasoning: {st.thought.strip()}\n" if st.thought.strip() else ""
        blocks.append({"type": "text", "text": f"[step {i + 1}]\n{thought}Tool call: {st.action.tool} "
                                               f"{json.dumps(st.action.args, ensure_ascii=False)}\nOutput:\n{obs}"})
    return blocks


def teacher_task(demo: Demo, prefix: list[Step]) -> list[dict]:
    head = {"type": "text", "text": build_task(demo.problem_statement, demo.test_cmd)}
    tail = {"type": "text", "text": RELABEL_INSTRUCTION if prefix else
            "No steps have been taken yet. Reply with the single best first tool call."}
    return [head, *transcript_blocks(prefix), tail]


def teacher_action(teacher_factory: Callable[[], object], demo: Demo, prefix: list[Step],
                   options: AgentOptions) -> tuple[str, Action, str, str | None]:
    """-> (thought, action, reasoning, text). Policies that expose their channels separately set
    last_reasoning / last_text; otherwise reasoning is "" and text is None (unknown)."""
    pol = teacher_factory()
    pol.reset(options.system_prompt(), teacher_task(demo, prefix))   # content blocks work for both APIs
    thought, action = pol.act(None)
    return thought, action, getattr(pol, "last_reasoning", "") or "", getattr(pol, "last_text", None)


# ---- agreement --------------------------------------------------------------------------------
def _norm_cmd(cmd: str) -> str:
    try:
        return " ".join(shlex.split(cmd))
    except ValueError:
        return " ".join(cmd.split())


def actions_agree(a: Action | None, b: Action | None) -> bool:
    """Loose equivalence: same tool and the same target (path / command / edit)."""
    if a is None or b is None or a.tool != b.tool:
        return False
    x, y = a.args or {}, b.args or {}
    if a.tool == "bash":
        return _norm_cmd(str(x.get("command", ""))) == _norm_cmd(str(y.get("command", "")))
    if a.tool in ("str_replace", "insert", "create_file"):
        keys = {"str_replace": ("path", "new_str"), "insert": ("path", "new_str"), "create_file": ("path",)}[a.tool]
        return all(str(x.get(k, "")).strip() == str(y.get(k, "")).strip() for k in keys)
    if a.tool in ("view_file", "find_files", "search", "undo_edit"):
        return x.get("path") == y.get("path") and x.get("pattern") == y.get("pattern")
    return True   # submit, run_tests


# ---- relabeling ---------------------------------------------------------------------------------
def states_to_relabel(rollout: Demo, every: int = 1, max_relabels: int | None = None) -> list[int]:
    n = len(rollout.trajectory)
    idx = list(range(0, n, max(1, every)))
    if max_relabels is not None and len(idx) > max_relabels:   # spread evenly over the episode
        idx = sorted({idx[round(k * (len(idx) - 1) / max(1, max_relabels - 1))] for k in range(max_relabels)})
    return idx


def relabel_rollout(rollout: Demo, teacher_factory: Callable[[], object], every: int = 1,
                    max_relabels: int | None = None, options: AgentOptions | None = None) -> list[Relabel]:
    options = options or student_options()
    out = []
    for t in states_to_relabel(rollout, every, max_relabels):
        prefix = rollout.trajectory[:t]
        st = rollout.trajectory[t] if t < len(rollout.trajectory) else None
        thought, action, reasoning, text = teacher_action(teacher_factory, rollout, prefix, options)
        out.append(Relabel(t, st.thought if st else "", st.action if st else None, thought, action,
                           actions_agree(st.action if st else None, action), reasoning, text))
    return out


# ---- outputs ------------------------------------------------------------------------------------
def to_sft_records(rollout: Demo, relabels: list[Relabel], cfg: StudentServingConfig | None = None,
                   agree_weight: float = 0.3, include_agreeing: bool = True, signal=None) -> list[dict]:
    """Student-format conversation up to each relabeled state, then the teacher's action as the
    only trained turn (loss_mask True on it alone). Same rendering as sft-mix records.
    `signal` (teacher_signal.SignalPlan) decides what happens to the teacher's reasoning: action
    mode trains on the visible text + tool call only; reasoning mode also trains on the reasoning,
    placed where the student keeps its own and weighted by plan.reasoning_weight."""
    cfg = cfg or StudentServingConfig()
    tools = sft.tools_as_functions()
    sys_prompt = cfg.agent_options().system_prompt()
    recs = []
    for r in relabels:
        if r.agree and not include_agreeing:
            continue
        prefix_demo = replace(rollout, trajectory=rollout.trajectory[:r.step])
        msgs = sft.render_messages(prefix_demo, sys_prompt, reasoning_field=cfg.keep_reasoning)
        call = [{"id": f"call_{r.step}", "type": "function", "function": {
            "name": r.teacher_action.tool, "arguments": json.dumps(r.teacher_action.args)}}]
        if signal is not None and r.teacher_text is not None:
            from .teacher_signal import shape_target

            target = {**shape_target(r.teacher_reasoning, r.teacher_text, signal), "tool_calls": call}
        else:   # channels unknown: previous behaviour (whole thought as reasoning or content)
            target = {"role": "assistant", "content": "" if cfg.keep_reasoning else r.teacher_thought,
                      "tool_calls": call}
            if cfg.keep_reasoning and r.teacher_thought:
                target["reasoning_content"] = r.teacher_thought
        msgs = apply_context_budget(msgs, tools, cfg) + [target]
        parent = rollout.metadata.get("parent_id") or rollout.instance_id
        recs.append({
            "id": f"{parent}#s{rollout.metadata.get('sample_idx', 0)}@{r.step}",
            "messages": msgs, "tools": tools,
            "loss_mask": [False] * (len(msgs) - 1) + [True],
            "weight": agree_weight if r.agree else 1.0,
            "source": "onpolicy-distill",
            "meta": {"parent_id": parent, "state": r.step, "agree": r.agree,
                     "student_resolved": rollout.metadata.get("resolved"),
                     "signal": signal.mode if signal is not None else None},
        })
    return recs


def to_step_pairs(rollout: Demo, relabels: list[Relabel], cfg: StudentServingConfig | None = None,
                  signal=None) -> list[dict]:
    """DPO pairs at disagreement states: chosen = teacher action, rejected = student action."""
    cfg = cfg or StudentServingConfig()
    pairs = []
    for rec, r in zip(to_sft_records(rollout, relabels, cfg, signal=signal), relabels):
        if r.agree or r.student_action is None:
            continue
        rejected = {"role": "assistant", "content": "" if cfg.keep_reasoning else r.student_thought,
                    "tool_calls": [{"id": f"call_{r.step}", "type": "function", "function": {
                        "name": r.student_action.tool, "arguments": json.dumps(r.student_action.args)}}]}
        if cfg.keep_reasoning and r.student_thought:
            rejected["reasoning_content"] = r.student_thought
        pairs.append({"id": rec["id"], "type": "onpolicy-step", "prompt": rec["messages"][:-1],
                      "chosen": [rec["messages"][-1]], "rejected": [rejected], "tools": rec["tools"],
                      "meta": rec["meta"]})
    return pairs


def select_rollouts(rollouts: list[Demo], only_unresolved: bool = False) -> list[Demo]:
    return [r for r in rollouts if r.trajectory and not (only_unresolved and r.metadata.get("resolved"))]


def estimate(rollouts: list[Demo], every: int, max_relabels: int | None, chars_per_token: float = 3.5,
             price_in: float = 4.0, price_out: float = 20.0, out_tokens: int = 400) -> dict:
    """Upper-bound cost (no prompt-cache discount) of relabeling `rollouts` with a Claude Opus-priced teacher."""
    calls, in_tok = 0, 0
    for r in rollouts:
        for t in states_to_relabel(r, every, max_relabels):
            calls += 1
            chars = len(r.problem_statement) + sum(len(b["text"]) for b in transcript_blocks(r.trajectory[:t]))
            in_tok += math.ceil(chars / chars_per_token) + 1500   # + system prompt and tool schemas
    return {"episodes": len(rollouts), "teacher_calls": calls, "input_tokens_upper": in_tok,
            "output_tokens_est": calls * out_tokens,
            "cost_upper_usd": round((in_tok * price_in + calls * out_tokens * price_out) / 1e6, 2)}


def onpolicy_distill(rollouts: list[Demo], teacher_factory: Callable[[], object], every: int = 1,
                     max_relabels: int | None = None, workers: int = 1, cfg: StudentServingConfig | None = None,
                     agree_weight: float = 0.3, signal=None) -> tuple[list[dict], list[dict], dict]:
    cfg = cfg or StudentServingConfig()
    options = cfg.agent_options()

    def one(r: Demo):
        rel = relabel_rollout(r, teacher_factory, every, max_relabels, options)
        return to_sft_records(r, rel, cfg, agree_weight, signal=signal), to_step_pairs(r, rel, cfg, signal), rel

    results = pmap(one, rollouts, workers)
    sft_recs = [x for s, _, _ in results for x in s]
    pairs = [x for _, p, _ in results for x in p]
    rels = [x for _, _, rel in results for x in rel]
    stats = {"episodes": len(rollouts), "relabels": len(rels), "agreement": (sum(r.agree for r in rels) / len(rels))
             if rels else None, "sft_records": len(sft_recs), "dpo_pairs": len(pairs)}
    return sft_recs, pairs, stats


# ---- CLI ------------------------------------------------------------------------------------------
def cmd_onpolicy(a) -> int:
    rollouts = select_rollouts(load_demos(a.rollouts), a.only_unresolved)
    if a.instance_weights:   # spend teacher calls only where the student doesn't already succeed
        from .difficulty import keep_for_teacher, load_weights

        w, before = load_weights(a.instance_weights), len(rollouts)
        rollouts = [r for r in rollouts
                    if keep_for_teacher(str(r.metadata.get("parent_id") or r.instance_id), w, a.max_pass_rate)]
        print(f"instance weights: kept {len(rollouts)}/{before} episodes with student pass rate <= {a.max_pass_rate}",
              file=sys.stderr)
    try:
        check_not_excluded(rollouts, load_exclude_ids(a.exclude_ids), "on-policy distillation")
    except ExcludedInstanceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    est = estimate(rollouts, a.every, a.max_relabels)
    print(json.dumps({"plan": est}), file=sys.stderr)
    if a.dry_run:
        print("dry run: no teacher calls made")
        return 0
    from ..policies import make_policy

    cfg = StudentServingConfig.load(a.config) if Path(a.config).exists() else StudentServingConfig()
    from .teacher_signal import default_caps, load_caps, select_mode

    t_caps, s_caps = load_caps(a.config)
    if t_caps.name != a.teacher:
        t_caps = default_caps(a.teacher)
    signal = select_mode(t_caps, s_caps, a.signal if a.signal in ("action", "reasoning") else None)
    if signal.mode not in ("action", "reasoning"):   # token modes have their own data path; relabels use reasoning
        signal = select_mode(t_caps, s_caps, "reasoning" if t_caps.reasoning != "none" else "action")
    if a.reasoning_weight is not None:
        signal.reasoning_weight = a.reasoning_weight
    print(f"signal: {signal.mode} (reasoning -> {signal.reasoning_target}, weight {signal.reasoning_weight})",
          file=sys.stderr)
    recs, pairs, stats = onpolicy_distill(rollouts, lambda: make_policy(a.teacher), a.every, a.max_relabels,
                                          a.workers, cfg, a.agree_weight, signal)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
    if a.pairs_out:
        Path(a.pairs_out).write_text("".join(json.dumps(p) + "\n" for p in pairs))
    print(json.dumps(stats))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("onpolicy-distill",
                              help="on-policy distillation: teacher relabels states the student reached")
    p.add_argument("--rollouts", required=True, help="student episodes (repo-fix eval --save-rollouts)")
    p.add_argument("--teacher", default="claude-opus-5-5")
    p.add_argument("--out", default="runs/onpolicy/sft.jsonl", help="SFT records (sft-mix record format)")
    p.add_argument("--pairs-out", default=None, help="step-level DPO pairs at disagreement states")
    p.add_argument("--every", type=int, default=2, help="relabel every k-th state")
    p.add_argument("--max-relabels", type=int, default=8, help="per episode cap (spread evenly)")
    p.add_argument("--only-unresolved", action="store_true", help="relabel only episodes the student failed")
    p.add_argument("--agree-weight", type=float, default=0.3, help="loss weight where student already agreed")
    p.add_argument("--signal", choices=["auto", "action", "reasoning"], default="auto",
                   help="train on the teacher's reasoning too (reasoning) or its action only; auto = from distill-signal")
    p.add_argument("--reasoning-weight", type=float, default=None,
                   help="override the reasoning loss weight (default 1.0 for full reasoning, 0.5 for summarized)")
    p.add_argument("--instance-weights", default=None, help="difficulty-weights JSON (focus on low pass rates)")
    p.add_argument("--max-pass-rate", type=float, default=0.75,
                   help="with --instance-weights: relabel only instances the student solves at most this often")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--exclude-ids", default=None, help="frozen test manifest")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the call/cost estimate and stop")
    p.set_defaults(fn=cmd_onpolicy)
