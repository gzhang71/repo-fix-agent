"""§3a SFT mix: engineer + teacher trajectories (+ aux tasks) rendered exactly like inference.

* Rendering: `sft.render_messages` (the student's system prompt from its AgentOptions, build_task,
  tool schemas and reasoning field exactly as served), then
  `policies.apply_context_budget` with the student's `StudentServingConfig`: the ONE truncation
  policy shared with `LocalModelPolicy`, so training contexts equal inference contexts.
* Filters: engineer demos need `metadata.replay_ok` not False; teacher rollouts need
  `metadata.resolved` True.
* `backfill_thoughts`: teacher writes rationales for empty/terse thoughts given only what the agent
  had seen; leaky backfills (gold-patch lines before the file was viewed, hidden test names) are rejected.
* `collapse_noops`: drop duplicate read-only commands and immediately-retried failed commands, kept
  only if the cleaned trajectory still reproduces `demo.patch` when re-executed.
* `check_prompt_consistency`: diff the SFT prefix against what `LocalModelPolicy` actually sends.

Shared helpers used by the sibling §3d/§3e modules: `load_exclude_ids`, `check_excluded`,
`instance_key`, `read_jsonl`, `write_jsonl`.
"""
from __future__ import annotations

import copy
import difflib
import json
import random
import re
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterable

from .. import sft
from ..agent import build_task, run_agent
from ..env import RepoEnv
from ..llm import TextLLM
from ..policies import LocalModelPolicy, StudentServingConfig, apply_context_budget, estimate_tokens
from ..schema import Action, Demo, Step, load_demos

CHARS_PER_TOKEN = 3.5


# ---- shared helpers (frozen-test guard, IO) ---------------------------------------------------
def instance_key(demo: Demo) -> str:
    """The source instance a record belongs to (rollouts point at it via metadata.parent_id)."""
    return str(demo.metadata.get("parent_id") or demo.instance_id)


def load_exclude_ids(path: str | Path | None) -> set[str]:
    """JSON manifest with an "instances" mapping/list (or a bare list), else newline-separated ids."""
    if not path:
        return set()
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")}
    if isinstance(data, dict):
        data = data.get("instances", data.get("ids", []))
    if isinstance(data, dict):
        return {str(k) for k in data}
    out: set[str] = set()
    for item in data or []:
        if isinstance(item, dict):
            item = item.get("instance_id") or item.get("id")
        if item:
            out.add(str(item))
    return out


def _ids_of(item: Any) -> set[str]:
    if isinstance(item, Demo):
        return {item.instance_id, instance_key(item)}
    if isinstance(item, dict):  # aux / SFT records
        meta = item.get("meta") or item.get("metadata") or {}
        return {str(x) for x in (item.get("instance_id"), meta.get("instance_id"), meta.get("parent_id")) if x}
    return set()


class ExcludedInstanceError(ValueError):
    """An input belongs to the frozen test set."""


def guard_frozen(fn):
    """Wrap a CLI handler: a frozen-test violation becomes a one-line error and exit code 2."""
    def wrapper(a) -> int:
        try:
            return fn(a)
        except ExcludedInstanceError as e:
            import sys
            print(f"error: {e}", file=sys.stderr)
            return 2
    return wrapper


def check_excluded(items: Iterable[Any], exclude_ids: set[str] | None, what: str = "input") -> None:
    """Raise if any demo/record (or its parent instance) is in the frozen-test exclusion set."""
    if not exclude_ids:
        return
    bad = sorted({i for it in items for i in _ids_of(it) if i in exclude_ids})
    if bad:
        raise ExcludedInstanceError(f"{what} contains {len(bad)} excluded (frozen-test) instance(s): {bad[:10]}")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def repo_path(demo: Demo, root: str | Path = ".") -> str:
    return demo.repo if "://" in demo.repo else str(Path(root) / demo.repo)


def load_serving_config(path: str | Path | None = "configs/models.yaml") -> StudentServingConfig:
    if path and Path(path).exists():
        return StudentServingConfig.load(path)
    return StudentServingConfig()


# ---- filters / rendering ----------------------------------------------------------------------
def keep_engineer(demo: Demo) -> bool:
    return demo.metadata.get("replay_ok") is not False and bool(demo.trajectory)


def keep_teacher(demo: Demo) -> bool:
    return demo.metadata.get("resolved") is True and bool(demo.trajectory)


def message_tokens(messages: list[dict]) -> int:
    return estimate_tokens(messages, [], CHARS_PER_TOKEN)


def render_record(demo: Demo, cfg: StudentServingConfig | None = None, source: str | None = None,
                  record_id: str | None = None) -> dict:
    cfg = cfg or StudentServingConfig()
    tools = sft.tools_as_functions()
    messages = apply_context_budget(
        sft.render_messages(demo, cfg.agent_options().system_prompt(), reasoning_field=cfg.keep_reasoning), tools, cfg)
    return {
        "id": record_id or demo.instance_id,
        "messages": messages,
        "tools": tools,
        "loss_mask": [m["role"] == "assistant" for m in messages],
        "weight": float(demo.metadata.get("weight", 1.0)),
        "source": source or demo.metadata.get("source") or "unknown",
    }


def aux_to_record(ex: dict) -> dict:
    meta = ex.get("meta") or {}
    return {"id": ex["id"], "messages": ex["messages"], "tools": [],
            "loss_mask": [m["role"] == "assistant" for m in ex["messages"]],
            "weight": float(meta.get("weight", 1.0)), "source": f"aux:{ex.get('task', 'unknown')}"}


# ---- thought backfill -------------------------------------------------------------------------
BACKFILL_SYSTEM = """You write the short rationale an engineer would have written right before a tool call
while fixing a bug. You are shown the issue, the steps taken so far (actions and their outputs) and the
NEXT action, which is fixed. Write one or two sentences explaining what the next action checks or
changes and why. Use ONLY information visible in the issue and the outputs shown so far: do not guess
file contents that have not been displayed, do not mention tests that have not been shown, and do not
mention any reference solution. Reply with the rationale only."""

_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS.sub("", s).lower()


def patch_files(patch: str) -> list[str]:
    """Files touched by a unified diff (post-image path; pre-image for deletions)."""
    files, old = [], None
    for ln in patch.splitlines():
        if ln.startswith("--- "):
            old = ln[4:].strip()
            old = old[2:] if old.startswith("a/") else old
        elif ln.startswith("+++ "):
            new = ln[4:].strip()
            path = old if new == "/dev/null" else (new[2:] if new.startswith("b/") else new)
            if path and path != "/dev/null" and path not in files:
                files.append(path)
    return files


def added_lines_by_file(patch: str, min_len: int = 8) -> dict[str, list[str]]:
    """Normalized `+` lines of a patch (len >= min_len after normalization), grouped by file."""
    out: dict[str, list[str]] = {}
    cur = None
    for ln in patch.splitlines():
        if ln.startswith("+++ "):
            p = ln[4:].strip()
            cur = p[2:] if p.startswith("b/") else p
            out.setdefault(cur, [])
        elif ln.startswith("+") and cur:
            n = _norm_text(ln[1:])
            if len(n) >= min_len:
                out[cur].append(n)
    return {k: v for k, v in out.items() if v}


def hidden_test_names(demo: Demo) -> set[str]:
    """Names that only exist in the hidden tests: F2P test ids, test functions and new test files."""
    names: set[str] = set()
    for tid in demo.FAIL_TO_PASS:
        last = tid.split("::")[-1].split("[")[0]
        if last:
            names.add(last)
    names.update(re.findall(r"^\+\s*(?:async\s+)?def\s+(test\w+)", demo.test_patch, re.M))
    names.update(re.findall(r"^\+\s*class\s+(Test\w+)", demo.test_patch, re.M))
    lines = demo.test_patch.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("--- /dev/null") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            stem = Path(lines[i + 1][4:].strip()).stem
            if len(stem) >= 6:
                names.add(stem)
    return {n for n in names if len(n) >= 6}


def _strip_dot(p: str) -> str:
    return p[2:] if p.startswith("./") else p


def files_referenced(step: Step, candidates: Iterable[str]) -> set[str]:
    """Which candidate files a step's action looked at (any string arg naming the path)."""
    blobs = [_strip_dot(v) for v in step.action.args.values() if isinstance(v, str)]
    blobs += [v for v in step.action.args.values() if isinstance(v, str)]
    return {f for f in candidates if any(f in b for b in blobs)}


def check_backfill(demo: Demo, step_idx: int, text: str) -> str | None:
    """Return a rejection reason, or None if the backfilled thought is acceptable."""
    if not text or not text.strip():
        return "empty"
    norm = _norm_text(text)
    gold = added_lines_by_file(demo.patch)
    viewed: set[str] = set()
    for st in demo.trajectory[:step_idx]:
        viewed |= files_referenced(st, gold)
    for f, lines in gold.items():
        if f not in viewed and any(ln in norm for ln in lines):
            return f"mentions gold-patch content of {f} before it was viewed"
    low = text.lower()
    for name in hidden_test_names(demo):
        if name.lower() in low:
            return f"mentions hidden test {name!r}"
    return None


def _clip(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[: n // 2] + "\n...[clipped]...\n" + s[-n // 2:]


def backfill_prompt(demo: Demo, step_idx: int) -> str:
    parts = [build_task(demo.problem_statement, demo.test_cmd), "", "Steps so far:"]
    if step_idx == 0:
        parts.append("(none)")
    for i, st in enumerate(demo.trajectory[:step_idx]):
        parts.append(f"[{i + 1}] action: {json.dumps({'tool': st.action.tool, 'args': st.action.args})}")
        parts.append(f"    output:\n{_clip(st.observation)}")
    nxt = demo.trajectory[step_idx].action
    parts += ["", f"NEXT action (fixed): {json.dumps({'tool': nxt.tool, 'args': nxt.args})}",
              "", "Rationale for the next action:"]
    return "\n".join(parts)


def backfill_thoughts(demo: Demo, llm: TextLLM, min_chars: int = 20) -> Demo:
    """Fill empty/terse thoughts via `llm`; rejected backfills keep the original thought."""
    out = copy.deepcopy(demo)
    accepted = rejected = 0
    for i, st in enumerate(out.trajectory):
        if len(st.thought.strip()) >= min_chars:
            continue
        try:
            text = (llm(BACKFILL_SYSTEM, backfill_prompt(demo, i)) or "").strip()
        except Exception:  # refusals / transient errors count as rejections
            text = ""
        if check_backfill(demo, i, text) is None:
            out.trajectory[i] = replace(st, thought=text)
            accepted += 1
        else:
            rejected += 1
    out.metadata = {**out.metadata, "backfill": {"accepted": accepted, "rejected": rejected}}
    return out


# ---- no-op collapse ---------------------------------------------------------------------------
_READ_ONLY_CMDS = {"ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "wc", "tree", "file",
                   "stat", "which", "less", "more", "echo", "nl"}
_READ_ONLY_GIT = {"status", "diff", "log", "show", "grep", "ls-files"}
_READ_ONLY_TOOLS = {"view_file", "search", "find_files"}
_EXIT_RE = re.compile(r"\[exit code (-?\d+)\]\s*$")


def is_read_only(step: Step) -> bool:
    if step.action.tool in _READ_ONLY_TOOLS:
        return True
    if step.action.tool != "bash":
        return False
    cmd = step.action.args.get("command", "")
    if re.search(r"(^|[^2])>|\btee\b|\bsed\s+-i|\brm\b|\bmv\b|\bcp\b", cmd):
        return False
    for part in re.split(r"&&|\|\||;|\|", cmd):
        try:
            words = shlex.split(part)
        except ValueError:
            return False
        if not words:
            continue
        if words[0] == "cd":
            continue
        if words[0] == "git" and len(words) > 1 and words[1] in _READ_ONLY_GIT:
            continue
        if words[0] not in _READ_ONLY_CMDS:
            return False
    return True


def _is_test_run(step: Step) -> bool:
    return step.action.tool == "run_tests" or (
        step.action.tool == "bash" and "pytest" in step.action.args.get("command", ""))


def step_failed(step: Step) -> bool:
    if step.observation.startswith("Error:"):
        return True
    if _is_test_run(step):  # a red test run is information, not a typo
        return False
    m = _EXIT_RE.search(step.observation)
    return bool(m) and int(m.group(1)) != 0


def _is_retry(failed: Step, nxt: Step) -> bool:
    if failed.action.tool != nxt.action.tool:
        return False
    if failed.action.tool == "bash":
        a, b = failed.action.args.get("command", ""), nxt.action.args.get("command", "")
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6
    return failed.action.args.get("path") == nxt.action.args.get("path")


def find_noops(trajectory: list[Step]) -> list[int]:
    drop = []
    for i, st in enumerate(trajectory):
        prev = trajectory[i - 1] if i else None
        nxt = trajectory[i + 1] if i + 1 < len(trajectory) else None
        if prev is not None and is_read_only(st) and st.action == prev.action:
            drop.append(i)
        elif nxt is not None and step_failed(st) and _is_retry(st, nxt):
            drop.append(i)
    return drop


def normalize_patch(p: str) -> str:
    keep = [ln.rstrip() for ln in p.splitlines() if not ln.startswith("index ")]
    return "\n".join(ln for ln in keep if ln).strip()


def patches_equivalent(a: str, b: str) -> bool:
    return normalize_patch(a) == normalize_patch(b)


def replay(demo: Demo, steps: list[Step], root: str | Path = ".", backend: str = "auto") -> tuple[list[Step], str]:
    """Re-execute `steps`' actions; return steps with fresh observations and the final diff."""
    out = []
    with RepoEnv(repo_path(demo, root), demo.base_commit, backend=backend) as env:
        for st in steps:
            out.append(Step(st.thought, st.action, env.execute(st.action)))
            if env.done:
                break
        return out, env.diff()


def collapse_noops(demo: Demo, root: str | Path = ".", backend: str = "auto") -> Demo:
    drop = set(find_noops(demo.trajectory))
    if not drop:
        return demo
    cleaned = [st for i, st in enumerate(demo.trajectory) if i not in drop]
    try:
        new_steps, diff = replay(demo, cleaned, root, backend)
    except Exception:
        return demo
    if not patches_equivalent(diff, demo.patch):
        return demo
    out = copy.deepcopy(demo)
    out.trajectory = new_steps
    out.metadata = {**out.metadata, "collapsed_steps": len(demo.trajectory) - len(new_steps)}
    return out


# ---- mix --------------------------------------------------------------------------------------
def _counts(n_e: int, n_t: int, ratio: tuple[float, float]) -> tuple[int, int]:
    re_, rt = ratio
    if n_t == 0 or rt <= 0:
        return n_e, 0
    if n_e == 0 or re_ <= 0:
        return 0, n_t
    k = min(n_e / re_, n_t / rt)
    return min(n_e, max(1, round(k * re_))), min(n_t, max(1, round(k * rt)))


def build_sft_mix(engineer: list[Demo], teacher: list[Demo] | None = None, aux: list[dict] | None = None,
                  ratio: tuple[float, float] = (1, 3), aux_frac: float = 0.2, seed: int = 0,
                  cfg: StudentServingConfig | None = None, exclude_ids: set[str] | None = None,
                  edge_case_weight: float = 1.0) -> list[dict]:
    """ratio = (engineer, teacher) by trajectory count; aux capped at aux_frac of total tokens.
    `edge_case_weight` multiplies the loss weight of trajectories on instances whose hidden tests go
    beyond the symptom in the issue (metadata.hidden_beyond_issue > 0, or several FAIL_TO_PASS)."""
    teacher, aux = teacher or [], aux or []
    check_excluded(engineer, exclude_ids, "engineer demos")
    check_excluded(teacher, exclude_ids, "teacher rollouts")
    check_excluded(aux, exclude_ids, "aux examples")
    cfg = cfg or StudentServingConfig()
    rng = random.Random(seed)
    eng = [d for d in engineer if keep_engineer(d)]
    tea = [d for d in teacher if keep_teacher(d)]
    n_e, n_t = _counts(len(eng), len(tea), ratio)
    eng, tea = rng.sample(eng, n_e), rng.sample(tea, n_t)

    records, seen = [], {}
    for d, default_src in [(d, "engineer") for d in eng] + [(d, "teacher") for d in tea]:
        rid = d.instance_id
        seen[rid] = seen.get(rid, 0) + 1
        if seen[rid] > 1:
            rid = f"{rid}#{seen[rid] - 1}"
        rec = render_record(d, cfg, d.metadata.get("source") or default_src, rid)
        beyond = d.metadata.get("hidden_beyond_issue")
        if edge_case_weight != 1.0 and ((beyond or 0) > 0 or (beyond is None and len(d.FAIL_TO_PASS) > 1)):
            rec["weight"] *= edge_case_weight
            rec["edge_case"] = True
        records.append(rec)

    traj_tokens = sum(message_tokens(r["messages"]) for r in records)
    if aux and aux_frac > 0:
        cap = traj_tokens * aux_frac / (1 - aux_frac) if aux_frac < 1 else float("inf")
        used = 0
        for ex in rng.sample(aux, len(aux)):
            rec = aux_to_record(ex)
            t = message_tokens(rec["messages"])
            if used + t > cap:
                continue
            records.append(rec)
            used += t
    rng.shuffle(records)
    return records


def mix_stats(records: list[dict]) -> dict:
    by_src: dict[str, dict] = {}
    for r in records:
        key = r["source"].split(":")[0] if r["source"].startswith("aux:") else r["source"]
        s = by_src.setdefault(key, {"n": 0, "tokens": 0})
        s["n"] += 1
        s["tokens"] += message_tokens(r["messages"])
    return {"n": len(records), "by_source": by_src}


# ---- train/inference prompt consistency -------------------------------------------------------
class _CaptureClient:
    """Fake OpenAI-compatible client: replays the demo's actions and records every request."""

    def __init__(self, steps: list[Step], reasoning: bool = False):
        self.requests: list[dict] = []
        self._steps = list(steps)
        self._reasoning = reasoning   # answer like a reasoning model: thought in reasoning_content
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(copy.deepcopy(kw))
        i = len(self.requests) - 1
        st = self._steps[i] if i < len(self._steps) else Step("", Action("submit", {}), "")
        tc = NS(id=f"call_{i}", function=NS(name=st.action.tool, arguments=json.dumps(st.action.args)))
        msg = (NS(content="", reasoning_content=st.thought, tool_calls=[tc]) if self._reasoning
               else NS(content=st.thought, tool_calls=[tc]))
        return NS(model="consistency-check", choices=[NS(message=msg)])


class _ReplayEnv:
    """Stands in for RepoEnv inside run_agent: returns the demo's recorded observations."""

    def __init__(self, steps: list[Step], n: int):
        self._obs = [s.observation for s in steps]
        self._n, self._i = n, 0

    @property
    def done(self) -> bool:
        return self._i >= self._n

    def execute(self, action) -> str:
        obs = self._obs[self._i] if self._i < len(self._obs) else ""
        self._i += 1
        return obs

    def diff(self) -> str:
        return ""


def check_prompt_consistency(demo: Demo, cfg: StudentServingConfig | None = None, steps: int = 2) -> list[str]:
    """Run `LocalModelPolicy` through the real `run_agent` loop (fake server, recorded observations)
    and compare each request with the budgeted SFT rendering of the same prefix.
    Returns a list of mismatch descriptions; empty means train == inference."""
    cfg = cfg or StudentServingConfig()
    n = max(1, min(steps, len(demo.trajectory)))
    client = _CaptureClient(demo.trajectory, reasoning=cfg.keep_reasoning)
    run_agent(LocalModelPolicy(cfg, client=client), _ReplayEnv(demo.trajectory, n),
              demo.problem_statement, demo.test_cmd, max_steps=n)
    rendered = sft.render_messages(demo, cfg.agent_options().system_prompt(), reasoning_field=cfg.keep_reasoning)
    tools = sft.tools_as_functions()
    problems = []
    if not client.requests:
        return ["policy sent no request"]
    first = client.requests[0]
    if first["messages"][0] != {"role": "system", "content": cfg.agent_options().system_prompt()} \
            or rendered[0] != first["messages"][0]:
        problems.append("system prompt differs")
    if rendered[1] != first["messages"][1]:
        problems.append("user task differs")
    for k, req in enumerate(client.requests):
        if req["tools"] != tools:
            problems.append(f"request {k}: tool schemas differ")
        want = apply_context_budget(rendered[: 2 + 2 * k], tools, cfg)
        if req["messages"] != want:
            problems.append(f"request {k}: message prefix differs from the SFT rendering")
    return problems


# ---- CLI --------------------------------------------------------------------------------------
def _parse_ratio(s: str) -> tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def cmd_sft_mix(a) -> int:
    exclude = load_exclude_ids(a.exclude_ids)
    cfg = load_serving_config(a.config)
    engineer = load_demos(a.engineer)
    teacher = [d for path in (a.teacher or []) for d in load_demos(path)]
    aux = read_jsonl(a.aux) if a.aux else []
    check_excluded(engineer + teacher, exclude, "inputs")
    check_excluded(aux, exclude, "aux examples")
    if a.collapse_noops:
        engineer = [collapse_noops(d, a.root) if keep_engineer(d) else d for d in engineer]
    if a.backfill_model:
        from ..llm import make_text_llm

        llm = make_text_llm(a.backfill_model)
        engineer = [backfill_thoughts(d, llm) if keep_engineer(d) else d for d in engineer]
    records = build_sft_mix(engineer, teacher, aux, _parse_ratio(a.ratio), a.aux_frac, a.seed, cfg, exclude,
                            edge_case_weight=a.edge_case_weight)
    for path in a.extra_records or []:   # pre-rendered records (e.g. onpolicy-distill): kept as-is
        extra = read_jsonl(path)
        check_excluded(extra, exclude, path)   # records carry meta.parent_id
        records += extra
    write_jsonl(records, a.out)
    print(json.dumps(mix_stats(records)))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("sft-mix", help="§3a: build the SFT mix (engineer + teacher + aux)")
    p.add_argument("--engineer", required=True)
    p.add_argument("--teacher", nargs="+", default=None)
    p.add_argument("--aux", default=None)
    p.add_argument("--out", default="runs/sft/sft_mix.jsonl")
    p.add_argument("--ratio", default="1:3", help="engineer:teacher by trajectory count")
    p.add_argument("--extra-records", action="append", default=None,
                   help="JSONL of already-rendered records to append unchanged (repeatable), e.g. onpolicy-distill output")
    p.add_argument("--edge-case-weight", type=float, default=1.0,
                   help="loss-weight multiplier for instances whose hidden tests go beyond the issue's example "
                        "(the student's main failure mode; try 2.0)")
    p.add_argument("--aux-frac", type=float, default=0.2, help="max share of total tokens for aux tasks")
    p.add_argument("--backfill-model", default=None, help="teacher model for thought backfill")
    p.add_argument("--collapse-noops", action="store_true", help="drop no-op steps (re-executes demos)")
    p.add_argument("--root", default=".", help="dir that demo.repo paths are relative to")
    p.add_argument("--config", default="configs/models.yaml", help="student.serving context budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-ids", default=None)
    p.set_defaults(fn=guard_frozen(cmd_sft_mix))
