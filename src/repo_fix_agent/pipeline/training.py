"""Student training: dataset export, Fireworks deployment, and the end-to-end round driver.

  repo-fix export-chat-ft runs/train/r1/sft.jsonl --out runs/train/r1/chat_ft.jsonl
  repo-fix fireworks-deploy --model accounts/<acct>/models/<id>          # dry run; add --yes to create
  repo-fix student-train --round r1                                      # print the plan + costs
  repo-fix student-train --round r1 --run distill --yes                  # run one (paid) stage

Where training runs: Fireworks' managed SFT/DPO stopped accepting new jobs on 2026-11-09 and
dataset uploads are disabled; new training goes through their Training API (Python SDK reading
local JSONL, https://docs.fireworks.ai/fine-tuning/training-api/introduction). The exported
chat-FT JSONL here follows the documented Fireworks/OpenAI chat schema (messages, tools,
per-message `weight`, `reasoning_content`), so it feeds that SDK or the bundled
`scripts/train_lora.py` (transformers + PEFT, runs on any GPU box: SFT, weighted, or GRPO-style
policy-gradient steps on `rl-rollouts` batches). The fine-tuned model is then served from a
Fireworks on-demand deployment (this student isn't serverless-LoRA capable) or vLLM, and
`configs/models.yaml` student.serving.model_name is pointed at it.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .sft_data import read_jsonl

FIREWORKS_API = "https://api.fireworks.ai/v1"
MIN_EXAMPLES, MAX_EXAMPLES = 3, 3_000_000   # Fireworks documented limits


# ---- dataset export --------------------------------------------------------------------------
def to_chat_ft(record: dict) -> dict:
    """sft-mix / onpolicy record -> chat fine-tuning example. The record's loss_mask and weight
    become per-message `weight` (0 = context only); Fireworks has no example-level weight."""
    w = float(record.get("weight", 1.0))
    mask = record.get("loss_mask") or [m.get("role") == "assistant" for m in record["messages"]]
    msgs = []
    for m, train in zip(record["messages"], mask):
        # sub-message reasoning weights are a train_lora.py feature; the chat-FT schema has one
        # weight per message, so they are dropped here (the whole message keeps the message weight)
        m = {k: v for k, v in m.items() if k not in ("reasoning_weight", "reasoning_text")}
        if m.get("role") == "assistant":
            m["weight"] = round(w, 4) if train else 0
        msgs.append(m)
    out = {"messages": msgs}
    if record.get("tools"):
        out["tools"] = record["tools"]
    return out


def validate_chat_ft(examples: list[dict]) -> list[str]:
    problems = []
    if not MIN_EXAMPLES <= len(examples) <= MAX_EXAMPLES:
        problems.append(f"{len(examples)} examples; Fireworks accepts {MIN_EXAMPLES}..{MAX_EXAMPLES}")
    for i, ex in enumerate(examples):
        msgs = ex.get("messages") or []
        if not any(m.get("role") == "assistant" and m.get("weight", 1) for m in msgs):
            problems.append(f"example {i}: no trained assistant turn")
        for m in msgs:
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                problems.append(f"example {i}: bad role {m.get('role')!r}")
            for tc in m.get("tool_calls") or []:
                try:
                    json.loads(tc["function"]["arguments"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    problems.append(f"example {i}: tool call arguments are not valid JSON")
        ids = {tc["id"] for m in msgs for tc in m.get("tool_calls") or [] if "id" in tc}
        for m in msgs:
            if m.get("role") == "tool" and m.get("tool_call_id") not in ids:
                problems.append(f"example {i}: tool result for unknown call {m.get('tool_call_id')!r}")
        if len(problems) > 50:
            problems.append("... (stopping)")
            break
    return problems


def export_chat_ft(records: list[dict], out: str | Path) -> tuple[int, list[str]]:
    examples = [to_chat_ft(r) for r in records]
    problems = validate_chat_ft(examples)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    return len(examples), problems


# ---- Fireworks deployment ----------------------------------------------------------------------
def fireworks_account() -> str:
    acct = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if acct:
        return acct
    req = urllib.request.Request(f"{FIREWORKS_API}/accounts",
                                 headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["accounts"][0]["name"].split("/", 1)[1]


def deployment_request(model: str, shape: str = "default", min_replicas: int = 0, max_replicas: int = 1) -> dict:
    """Body for POST /v1/accounts/{account}/deployments (live-merge a fine-tuned model)."""
    return {"baseModel": model, "deploymentShape": shape, "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas}


def create_deployment(model: str, shape: str = "default", min_replicas: int = 0, max_replicas: int = 1,
                      opener=None) -> dict:
    acct = fireworks_account()
    body = deployment_request(model, shape, min_replicas, max_replicas)
    req = urllib.request.Request(f"{FIREWORKS_API}/accounts/{acct}/deployments", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                                          "Content-Type": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Fireworks deployment failed ({e.code}): {e.read().decode(errors='replace')[:500]}")


# ---- round driver ------------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    what: str
    argv: list[str] | None          # repo-fix CLI args, or None for an external step
    paid: str = ""                  # "" = free; otherwise what it bills and an estimate
    external: str = ""              # command to run elsewhere (GPU box)
    needs: list[str] = field(default_factory=list)


def plan(round_dir: Path, splits: Path, prev_rollouts: str | None = None, samples: int = 4,
         teacher: str = "claude-opus-5-5", base_model_hf: str = "<HF id of the student base>",
         signal: str = "reasoning", teacher_weights: str = "<HF id of the teacher>",
         prev_round_dir: Path | None = None) -> list[Stage]:
    """`signal` (teacher_signal.select_mode) picks the distillation path: action / reasoning use
    on-policy relabels; topk adds teacher log-prob scoring + a tokenkl step; logits adds a GKD step
    against the teacher's weights."""
    r, m = round_dir, str(splits / "frozen_test.manifest.json")
    train, dev = str(splits / "train.jsonl"), str(splits / "dev.jsonl")
    rollouts = prev_rollouts or str(r / "student_rollouts.jsonl")
    relabel_signal = signal if signal in ("action", "reasoning") else "auto"
    rnd = r.name
    before_label, after_label = f"{rnd} dev before", f"{rnd} dev after"
    # anti-forgetting anchor on every training step (retention.py): KL to the base on retention records
    retain = f"--retain-data {r / 'retain.jsonl'} --retain-data {r / 'retain_general.jsonl'} --retain-beta 0.1"
    replay = ["--replay", str(prev_round_dir / "sft.jsonl")] if prev_round_dir else []
    stages = [
        Stage("augment", "synthetic bug-injected instances from train repos (edge-case tagged)",
              ["augment", train, "--out", str(r / "synthetic.jsonl"), "--repos-dir", str(r / "repos_synth"),
               "--max-per-demo", "2", "--exclude-ids", m]),
        Stage("student-rollouts", f"current student on train, {samples} samples/instance, saved as rollouts",
              ["eval", train, "--model", "student", "--samples", str(samples), "--temperature", "0.7",
               "--workers", "8", "--out", str(r / "student_eval.jsonl"), "--save-rollouts", rollouts],
              paid="Fireworks inference (student tokens; cheap)"),
        Stage("dev-before", "current student on the dev split, 3 samples: the reference for the forgetting gate",
              ["eval", dev, "--model", "student", "--samples", "3", "--temperature", "0.7", "--workers", "8",
               "--label", before_label], paid="Fireworks inference"),
        Stage("retain-set", "retention records: the student's own resolved episodes from before training",
              ["retain-set", "--rollouts", rollouts, "--exclude-ids", m, "--out", str(r / "retain.jsonl")],
              needs=["student-rollouts"]),
        Stage("retain-general", "retention records: the base student's answers to 32 general prompts",
              ["retain-set", "--general", "--yes", "--out", str(r / "retain_general.jsonl")],
              paid="Fireworks inference (32 short calls)"),
        Stage("weights", "per-instance student pass rate -> bucket (gap / learning / mastered / unsolved) "
                         "and SFT weight max(0.2, 1 - p)",
              ["difficulty-weights", str(r / "student_eval.jsonl"), "--out", str(r / "weights.json")],
              needs=["student-rollouts"]),
        Stage("distill", "teacher rollouts on train instances the student doesn't always solve (p <= 0.75), "
                         "rejection-sampled, under the student's prompt",
              ["distill", train, "--model", teacher, "--k", "2", "--hints", "--workers", "6", "--exclude-ids", m,
               "--instance-weights", str(r / "weights.json"), "--max-pass-rate", "0.75",
               "--out-resolved", str(r / "teacher.jsonl"), "--out-failed", str(r / "teacher_failed.jsonl")],
              paid="Claude: about $0.05 x 2 samples x focused instances (prompt caching on)", needs=["weights"]),
        Stage("distill-synthetic", "teacher rollouts on the synthetic instances",
              ["distill", str(r / "synthetic.jsonl"), "--model", teacher, "--k", "1", "--workers", "6",
               "--exclude-ids", m, "--out-resolved", str(r / "teacher_synth.jsonl"),
               "--out-failed", str(r / "teacher_synth_failed.jsonl")],
              paid="Claude: about $0.05 x synthetic instances", needs=["augment"]),
        Stage("onpolicy", "teacher relabels states the student reached (DAgger); + step DPO pairs",
              ["onpolicy-distill", "--rollouts", rollouts, "--teacher", teacher, "--exclude-ids", m,
               "--instance-weights", str(r / "weights.json"), "--max-pass-rate", "0.75", "--signal", relabel_signal,
               "--out", str(r / "onpolicy_sft.jsonl"), "--pairs-out", str(r / "onpolicy_pairs.jsonl")],
              paid="Claude: run with --dry-run first for the exact estimate", needs=["student-rollouts", "weights"]),
        Stage("aux", "verifiable auxiliary tasks (localization, test writing, repro, critique)",
              ["aux-tasks", train, "--out", str(r / "aux.jsonl"), "--exclude-ids", m]),
        Stage("prefs", "trajectory/step DPO pairs from student vs teacher episodes",
              ["prefs", "--rollouts", rollouts, str(r / "teacher.jsonl"), "--demos", train,
               "--out-dpo", str(r / "prefs.jsonl"), "--out-steps", str(r / "prefs_steps.jsonl"),
               "--balance-lengths", "--exclude-ids", m], needs=["student-rollouts", "distill"]),
        Stage("sft-mix", "engineer + teacher + student successes + on-policy + aux; weighted by difficulty "
                         "(1 - student pass rate, floor 0.2, capped per instance), edge cases 2x"
                         + ("; 20% replayed from " + prev_round_dir.name if prev_round_dir else ""),
              ["sft-mix", "--engineer", train, "--teacher", str(r / "teacher.jsonl"), str(r / "teacher_synth.jsonl"),
               "--aux", str(r / "aux.jsonl"), "--extra-records", str(r / "onpolicy_sft.jsonl"),
               "--student-rollouts", rollouts, "--instance-weights", str(r / "weights.json"),
               "--edge-case-weight", "2.0", "--exclude-ids", m, *replay, "--out", str(r / "sft.jsonl")],
              needs=["distill", "distill-synthetic", "aux", "onpolicy", "weights"]),
        Stage("export", "chat fine-tuning JSONL (Fireworks/OpenAI schema, per-message weights)",
              ["export-chat-ft", str(r / "sft.jsonl"), "--out", str(r / "chat_ft.jsonl")], needs=["sft-mix"]),
        Stage("train-sft", "LoRA SFT with the retention KL anchor, on a GPU box", None,
              paid="GPU hours", external=f"python scripts/train_lora.py --base {base_model_hf} "
                                         f"--data {r / 'chat_ft.jsonl'} --objective sft {retain} --out {r / 'lora_sft'}",
              needs=["export", "retain-set", "retain-general"]),
        Stage("train-dpo", "DPO on prefs + on-policy pairs from the SFT checkpoint", None, paid="GPU hours",
              external=f"python scripts/train_lora.py --base {base_model_hf} --adapter {r / 'lora_sft'} "
                       f"--objective dpo --data {r / 'prefs.jsonl'} --data {r / 'prefs_steps.jsonl'} "
                       f"--data {r / 'onpolicy_pairs.jsonl'} {retain} --out {r / 'lora_dpo'}",
              needs=["train-sft", "prefs"]),
        Stage("deploy", "serve the fine-tuned student (then set student.serving.model_name)",
              ["fireworks-deploy", "--model", "accounts/<acct>/models/<fine-tuned-id>"],
              paid="Fireworks on-demand GPU per hour (dry run unless --yes)", needs=["train-dpo"]),
        Stage("eval", "new student on the dev split, 3 samples (decide with mean pass@1)",
              ["eval", dev, "--model", "student", "--samples", "3", "--temperature", "0.7", "--workers", "8",
               "--label", after_label], paid="Fireworks inference", needs=["deploy"]),
        Stage("forgetting-gate", "previously solved dev instances must stay solved (<= 5% forgotten), accuracy "
                                 "must not drop, tool errors must not rise; on failure, interpolate the adapter "
                                 "toward the base (scripts/interpolate_lora.py --alpha 0.5) and re-evaluate",
              ["forgetting-check", "--before", before_label, "--after", after_label],
              needs=["eval", "dev-before"]),
        Stage("post-rollouts", f"fine-tuned student on train, {samples} samples/instance (pass rates moved)",
              ["eval", train, "--model", "student", "--samples", str(samples), "--temperature", "0.7",
               "--workers", "8", "--out", str(r / "post_eval.jsonl")],
              paid="Fireworks inference (student tokens)", needs=["deploy"]),
        Stage("rl-weights", "pass rates of the fine-tuned student -> RL curriculum (mixed-outcome instances)",
              ["difficulty-weights", str(r / "post_eval.jsonl"), "--out", str(r / "post_weights.json"),
               "--pass-rates-out", str(r / "pass_rates.json")], needs=["post-rollouts"]),
        Stage("rl-rollouts", "GRPO batches on instances the student solves only sometimes; shaped reward "
                             "(partial credit, broken-test penalty, verified-repro bonus); groups that all tie dropped",
              ["rl-rollouts", train, "--model", "student", "--group-size", "8", "--workers", "8",
               "--pass-rates", str(r / "pass_rates.json"), "--reward", "shaped", "--dynamic-sampling",
               "--exclude-ids", m, "--out", str(r / "rl_batch.jsonl")],
              paid="Fireworks inference (8 episodes per instance)", needs=["forgetting-gate", "rl-weights"]),
        Stage("train-rl", "clipped policy gradient (GRPO advantages, clip-higher, token-level loss) + KL + "
                          "retention anchor, 2 passes; then redeploy, re-run eval and the forgetting gate", None,
              paid="GPU hours", external=f"python scripts/train_lora.py --base {base_model_hf} --adapter "
                                         f"{r / 'lora_dpo'} --data {r / 'rl_batch.jsonl'} --objective pg "
                                         f"--loss-agg token --clip-low 0.2 --clip-high 0.28 --epochs 2 "
                                         f"{retain} --out {r / 'lora_rl'}", needs=["rl-rollouts"]),
    ]
    extra: list[Stage] = []
    if signal == "topk":
        extra = [
            Stage("token-distill", "same-tokenizer teacher scores the student's own episodes (per-token log-probs)",
                  ["token-distill", "--rollouts", rollouts, "--exclude-ids", m, "--yes",
                   "--out", str(r / "token_distill.jsonl")],
                  paid="teacher prompt tokens on the scoring endpoint (estimate printed first)",
                  needs=["student-rollouts"]),
            Stage("train-token", "token-level on-policy distillation (reverse KL + top-k forward KL)", None,
                  paid="GPU hours", external=f"python scripts/train_lora.py --base {base_model_hf} --adapter "
                                             f"{r / 'lora_dpo'} --objective tokenkl --data {r / 'token_distill.jsonl'} "
                                             f"{retain} --out {r / 'lora_tok'}", needs=["train-dpo", "token-distill"]),
        ]
    elif signal == "logits":
        extra = [
            Stage("render-rollouts", "student episodes as chat records for GKD",
                  ["token-distill", "--rollouts", rollouts, "--render-only", "--exclude-ids", m,
                   "--out", str(r / "rollout_records.jsonl")], needs=["student-rollouts"]),
            Stage("train-gkd", "GKD: full-distribution KL to the teacher's weights on the student's episodes", None,
                  paid="GPU hours (teacher + student loaded)",
                  external=f"python scripts/train_lora.py --base {base_model_hf} --adapter {r / 'lora_dpo'} "
                           f"--objective gkd --teacher-base {teacher_weights} --data {r / 'rollout_records.jsonl'} "
                           f"{retain} --out {r / 'lora_gkd'}", needs=["train-dpo", "render-rollouts"]),
        ]
    if extra:
        at = next(i for i, st in enumerate(stages) if st.name == "train-dpo") + 1
        stages[at:at] = extra
        deploy = next(st for st in stages if st.name == "deploy")
        deploy.needs = [extra[-1].name]
    return stages


def format_plan(stages: list[Stage]) -> str:
    lines = []
    for i, s in enumerate(stages, 1):
        cost = f"PAID: {s.paid}" if s.paid else "free"
        lines.append(f"{i:2}. {s.name:18} [{cost}]\n    {s.what}")
        if s.argv:
            lines.append("    repo-fix " + " ".join(shlex.quote(a) for a in s.argv))
        if s.external:
            lines.append(f"    (run on a GPU box) {s.external}")
        if s.needs:
            lines.append(f"    after: {', '.join(s.needs)}")
    return "\n".join(lines)


# ---- CLI ------------------------------------------------------------------------------------------
def cmd_export(a) -> int:
    n, problems = export_chat_ft(read_jsonl(a.records), a.out)
    print(f"wrote {n} examples to {a.out}")
    for p in problems[:20]:
        print(f"  problem: {p}", file=sys.stderr)
    return 1 if problems else 0


def cmd_deploy(a) -> int:
    body = deployment_request(a.model, a.shape, a.min_replicas, a.max_replicas)
    print(json.dumps({"POST": "/v1/accounts/{account}/deployments", "body": body}, indent=2))
    if not a.yes:
        print("dry run: nothing created (on-demand deployments bill per GPU hour). Re-run with --yes.")
        return 0
    d = create_deployment(a.model, a.shape, a.min_replicas, a.max_replicas)
    print(json.dumps(d, indent=2)[:2000])
    print(f"\nthen set configs/models.yaml student.serving.model_name: {a.model}")
    return 0


def cmd_student_train(a) -> int:
    from .teacher_signal import load_caps, select_mode

    round_dir = Path(a.runs) / a.round
    t_caps, s_caps = load_caps(a.config)
    try:
        sig = select_mode(t_caps, s_caps, None if a.signal == "auto" else a.signal)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stages = plan(round_dir, Path(a.splits), a.rollouts, a.samples, a.teacher, a.base_model, sig.mode,
                  t_caps.weights or "<HF id of the teacher>",
                  Path(a.runs) / a.prev_round if a.prev_round else None)
    if not a.run:
        print(f"Student training round {a.round!r} (outputs in {round_dir}); nothing is run without --run.")
        print(f"Distillation signal: {sig.mode} (repo-fix distill-signal explains why)\n")
        print(format_plan(stages))
        return 0
    st = next((s for s in stages if s.name == a.run), None)
    if st is None:
        print(f"unknown stage {a.run!r}; one of: {', '.join(s.name for s in stages)}", file=sys.stderr)
        return 2
    if st.argv is None:
        print(f"{st.name} runs outside this tool:\n  {st.external}")
        return 0
    if st.paid and not a.yes:
        print(f"{st.name} costs money ({st.paid}). Re-run with --yes to proceed.\n  repo-fix {' '.join(st.argv)}")
        return 3
    round_dir.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "repo_fix_agent.cli", *st.argv]
    print("+ " + " ".join(shlex.quote(x) for x in argv[2:]), flush=True)
    return subprocess.call(argv)


def register(subparsers) -> None:
    e = subparsers.add_parser("export-chat-ft", help="SFT records -> chat fine-tuning JSONL (Fireworks/OpenAI)")
    e.add_argument("records")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)

    d = subparsers.add_parser("fireworks-deploy", help="deploy a fine-tuned student on Fireworks (dry run by default)")
    d.add_argument("--model", required=True, help="accounts/<acct>/models/<fine-tuned-id>")
    d.add_argument("--shape", default="default")
    d.add_argument("--min-replicas", type=int, default=0)
    d.add_argument("--max-replicas", type=int, default=1)
    d.add_argument("--yes", action="store_true", help="actually create the (billed) deployment")
    d.set_defaults(fn=cmd_deploy)

    t = subparsers.add_parser("student-train", help="plan / run one stage of a student training round")
    t.add_argument("--round", default="r1")
    t.add_argument("--runs", default="runs/train")
    t.add_argument("--splits", default="runs/splits", help="output dir of `repo-fix split`")
    t.add_argument("--rollouts", default=None, help="reuse existing student rollouts JSONL")
    t.add_argument("--samples", type=int, default=4)
    t.add_argument("--teacher", default="claude-opus-5-5")
    t.add_argument("--base-model", default="<HF id of the student base>", help="HF weights for scripts/train_lora.py")
    t.add_argument("--signal", default="auto", choices=["auto", "action", "reasoning", "topk", "logits"],
                   help="distillation path; auto = richest the teacher/student pair supports")
    t.add_argument("--config", default="configs/models.yaml")
    t.add_argument("--prev-round", default=None, help="earlier round (e.g. r1): replay 20%% of its SFT data")
    t.add_argument("--run", default=None, help="stage name to execute")
    t.add_argument("--yes", action="store_true", help="allow a paid stage")
    t.set_defaults(fn=cmd_student_train)
