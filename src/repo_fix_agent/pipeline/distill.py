"""§3b teacher-hinted trajectory generation + §3c rejection-sampled teacher distillation.

Flow per train instance (``distill``):

  1. k unhinted teacher rollouts in fresh ``RepoEnv``s, each graded with ``grader.grade``.
  2. If none resolved and hints are enabled, escalate the hint ladder 1 -> 4 (k samples per level),
     stopping at the first level with >= 1 resolved rollout.
       1 FAIL_TO_PASS test names
       2 + files changed by the gold patch
       3 + the full test_patch
       4 + the full gold patch, framed as a reference solution to reproduce the reasoning of
     The hint is a separate system-prompt block added by ``HintedPolicy``. The saved record only
     carries the trajectory/patch (whose prompt is the plain ``build_task``); ``metadata.hint_level``
     records which level was used.
  3. Filters on resolved rollouts: tests run after the first edit and before submit; hint-leakage
     heuristics (a)-(c) drop hinted rollouts and only flag unhinted ones; (d) down-weights hinted
     rollouts whose steps-to-first-edit is far below the unhinted distribution for the same
     patch-size bucket; per-instance dedup (same action sequence or same normalized patch) keeping
     at most ``max_per_instance``, shorter first.
  4. Unresolved rollouts are returned separately (inputs for §3e preferences and §3f verifier).

``mix_datasets`` mixes teacher and engineer records at a trajectory-count ratio with per-instance
caps. Model access is only through an injected ``policy_factory`` so everything runs offline in tests.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .. import policies
from ..agent import run_agent
from ..env import RepoEnv
from ..grader import GradeResult, grade
from ..schema import Action, Demo, Step, load_demos, save_demos
from .parallel import pmap

PolicyFactory = Callable[[], "policies.Policy"]
GradeFn = Callable[[Demo, str, Any], GradeResult]

MAX_HINT_LEVEL = 4
EDIT_TOOLS = {"str_replace", "create_file", "insert"}
EXPLORE_TOOLS = {"view_file", "search", "find_files", "run_tests"}

# Instance-level metadata carried from the source demo to its rollouts (trajectory-specific keys
# such as num_steps / files_changed / replay_ok are not carried over).
INSTANCE_META_KEYS = ("bug_type", "difficulty", "language", "public", "cluster_id", "patch_size",
                      "flaky", "split", "repo_group")

DROP_REASONS = ("unresolved", "no_tests_before_submit", "leak_unsurfaced_first_edit",
                "leak_no_exploration", "leak_thought_mentions", "duplicate", "over_cap")


class FrozenInstanceError(ValueError):
    """Raised when asked to run teacher rollouts / hints on an excluded (frozen dev/test) instance."""


# ---- hints -----------------------------------------------------------------------------------
HINT_HEADER = (
    "PRIVILEGED HINT (training-data generation only). Use it to steer your investigation, but work "
    "exactly as you would without it: locate the relevant code with search/view, read before you "
    "edit, reproduce the bug, verify with the tests. Never mention or quote this hint in your "
    "reasoning; the hint will not be available at inference time."
)


def files_changed(patch: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", patch, re.M):
        if m.group(2) not in out:
            out.append(m.group(2))
    if not out:  # bare unified diff without git headers
        for m in re.finditer(r"^\+\+\+ (?:b/)?(\S+)", patch, re.M):
            if m.group(1) != "/dev/null" and m.group(1) not in out:
                out.append(m.group(1))
    return out


def build_hint(demo: Demo, level: int) -> str:
    """Hint text for ladder ``level`` (1-4); cumulative. Level 0 -> empty string."""
    if level <= 0:
        return ""
    if level > MAX_HINT_LEVEL:
        raise ValueError(f"hint level must be 0-{MAX_HINT_LEVEL}, got {level}")
    parts = [HINT_HEADER, "Tests that must pass after the fix (FAIL_TO_PASS):\n"
             + "\n".join(f"  - {t}" for t in demo.FAIL_TO_PASS)]
    if level >= 2:
        parts.append("Files changed by the fix:\n" + "\n".join(f"  - {f}" for f in files_changed(demo.patch)))
    if level >= 3:
        parts.append("The hidden tests that will grade the fix (test_patch):\n```diff\n"
                     + demo.test_patch.rstrip() + "\n```")
    if level >= 4:
        parts.append(
            "A reference solution an engineer wrote is below. Do not just copy it: reproduce the "
            "reasoning that leads to it and explore the repository normally (find and read the code, "
            "reproduce the bug, then fix and verify).\n```diff\n" + demo.patch.rstrip() + "\n```")
    return "\n\n".join(parts)


class HintedPolicy:
    """Wraps a policy; ``reset`` appends the hint as a separate block to the system prompt.

    The agent loop records only (thought, action, observation), so the hint never reaches the saved
    record; callers tag it via ``metadata.hint_level``."""

    def __init__(self, inner, hint: str, level: int | None = None):
        self.inner, self.hint, self.level = inner, hint, level

    def reset(self, system: str, task: str) -> None:
        sys_prompt = f"{system}\n\n<hint>\n{self.hint}\n</hint>" if self.hint else system
        self.inner.reset(sys_prompt, task)

    def act(self, last_observation: str | None) -> tuple[str, Action]:
        return self.inner.act(last_observation)

    def preload(self, steps: list[Step]) -> None:
        self.inner.preload(steps)

    @property
    def served_models(self):
        return getattr(self.inner, "served_models", set())


# ---- trajectory analysis ----------------------------------------------------------------------
_TEST_CMD = re.compile(r"\bpytest\b|\bpython3?\s+-m\s+(pytest|unittest)\b|\bunittest\b|\btox\b|\bnox\b")
_BASH_EDIT = re.compile(r"\bsed\s+(-[a-zA-Z]*i|--in-place)|\bperl\s+-[a-z]*i|\btee\s|\bpatch\s+-|\bgit\s+apply\b"
                        r"|(?<![0-9&])>>?\s*[\w./-]+\.\w+")
_BASH_READ = re.compile(r"^\s*(grep|egrep|rg|ag|cat|head|tail|less|more|ls|find|tree|sed\s+-n|awk|wc|nl|file"
                        r"|git\s+(log|show|grep|diff|status|blame|ls-files)|python3?|pytest)\b")
_PATH_IN_CMD = re.compile(r"[\w./-]+\.\w+")
_MENTION = re.compile(r"reference (solution|fix|patch|implementation)|\bthe reference\b|\bgold (patch|fix)\b"
                      r"|\bhints?\b|\bprivileged\b|FAIL_TO_PASS", re.I)


def _clean(p: str) -> str:
    p = p.strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _is_edit(step: Step) -> bool:
    a = step.action
    if a.tool in EDIT_TOOLS:
        return not step.observation.startswith("Error:")
    return a.tool == "bash" and bool(_BASH_EDIT.search(str(a.args.get("command", ""))))


def _edit_path(step: Step) -> str | None:
    a = step.action
    if a.tool in EDIT_TOOLS:
        return _clean(str(a.args.get("path", ""))) or None
    m = [p for p in _PATH_IN_CMD.findall(str(a.args.get("command", ""))) if not p.startswith("/dev/")]
    return _clean(m[-1]) if m else None


def _is_exploration(step: Step) -> bool:
    a = step.action
    if a.tool in EXPLORE_TOOLS:
        return True
    if a.tool == "bash":
        cmd = str(a.args.get("command", ""))
        return not _BASH_EDIT.search(cmd) and any(_BASH_READ.match(seg) for seg in re.split(r"&&|\|\||;|\|", cmd))
    return False


def _is_test_run(step: Step) -> bool:
    a = step.action
    return a.tool == "run_tests" or (a.tool == "bash" and bool(_TEST_CMD.search(str(a.args.get("command", "")))))


def first_edit_index(traj: list[Step]) -> int | None:
    return next((i for i, s in enumerate(traj) if _is_edit(s)), None)


def ran_tests_before_submit(traj: list[Step]) -> bool:
    """A test command (bash pytest / python -m pytest / unittest, or run_tests) ran after the first
    edit and before submit. Without any edit, any test run before submit counts."""
    end = next((i for i, s in enumerate(traj) if s.action.tool == "submit"), len(traj))
    start = first_edit_index(traj[:end])
    start = 0 if start is None else start + 1
    return any(_is_test_run(s) for s in traj[start:end])


def _path_variants(path: str) -> list[str]:
    path = _clean(path)
    variants = [path, path.rsplit("/", 1)[-1]]
    stem = path.rsplit(".", 1)[0]
    if "/" in stem:
        variants.append(stem.replace("/", "."))
    return [v for v in variants if v]


def _mentions_path(text: str, path: str) -> bool:
    return any(v in text for v in _path_variants(path))


def _touches_path(step: Step, path: str) -> bool:
    a = step.action
    if _clean(str(a.args.get("path", ""))) == path:
        return True
    return a.tool == "bash" and _mentions_path(str(a.args.get("command", "")), path)


def patch_bucket(patch: str) -> str:
    n_files = len(files_changed(patch))
    n_hunks = len(re.findall(r"^@@", patch, re.M))
    if n_files == 0:
        return "empty"
    if n_files > 1:
        return "multi-file"
    return "1-file-1-hunk" if n_hunks <= 1 else "1-file-multi-hunk"


def _gold_added_lines(patch: str, min_len: int = 12) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    cur = None
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line)
        if m:
            cur = m.group(1)
            continue
        if cur and line.startswith("+") and len(line[1:].strip()) >= min_len:
            out[cur].append(line[1:].strip())
    return out


def leakage_flags(demo: Demo, traj: list[Step], patch: str) -> list[str]:
    """§3b heuristics (a)-(c). Returns flag names; empty list = clean.

    (a) leak_unsurfaced_first_edit: the first edited file was not surfaced (path, basename or dotted
        module) by the issue text or any observation before the agent first touched it.
    (b) leak_no_exploration: no exploration step (view/search/find/read-only bash) before the first
        edit while the final patch touches > 1 file.
    (c) leak_thought_mentions: a thought names a FAIL_TO_PASS test (not yet observed), the
        reference/hint, or quotes an added gold-patch line before that file was viewed.
    """
    flags: list[str] = []
    fe = first_edit_index(traj)
    if fe is not None:
        path = _edit_path(traj[fe])
        if path:
            first_touch = next((i for i, s in enumerate(traj[: fe + 1]) if _touches_path(s, path)), fe)
            seen = demo.problem_statement + "\n" + "\n".join(s.observation for s in traj[:first_touch])
            if traj[fe].action.tool == "create_file" and "/" in path:
                ok = _mentions_path(seen, path) or path.rsplit("/", 1)[0].rsplit("/", 1)[-1] in seen
            else:
                ok = _mentions_path(seen, path)
            if not ok:
                flags.append("leak_unsurfaced_first_edit")
        if not any(_is_exploration(s) for s in traj[:fe]) and len(files_changed(patch)) > 1:
            flags.append("leak_no_exploration")

    test_names = set()
    for t in demo.FAIL_TO_PASS:
        test_names.add(t)
        leaf = re.sub(r"\[.*\]$", "", t.split("::")[-1])
        if len(leaf) >= 6:
            test_names.add(leaf)
    gold = _gold_added_lines(demo.patch)
    viewed: set[str] = set()
    seen = demo.problem_statement
    for s in traj:
        th = s.thought or ""
        if _MENTION.search(th) or any(n in th and n not in seen for n in test_names):
            flags.append("leak_thought_mentions")
            break
        if any(ln in th for f, lines in gold.items() if f not in viewed for ln in lines):
            flags.append("leak_thought_mentions")
            break
        for f in gold:
            if _touches_path(s, f):
                viewed.add(f)
        seen += "\n" + s.observation
    return flags


def action_signature(traj: list[Step]) -> str:
    return json.dumps([[s.action.tool, s.action.args] for s in traj], sort_keys=True)


def normalize_patch(patch: str) -> str:
    keep = []
    for line in patch.splitlines():
        if line.startswith("index ") or not line.strip():
            continue
        if line.startswith("@@"):
            line = re.sub(r"^(@@ [^@]* @@).*", r"\1", line)
        keep.append(line.rstrip())
    return "\n".join(keep)


# ---- rollouts -----------------------------------------------------------------------------------
def _check_excluded(demos: Iterable[Demo], exclude_ids: set[str] | None) -> None:
    if not exclude_ids:
        return
    bad = sorted({d.instance_id for d in demos if d.instance_id in exclude_ids})
    if bad:
        raise FrozenInstanceError(
            f"refusing to run teacher rollouts on {len(bad)} excluded (frozen) instance(s): {bad[:10]}")


def _repo_path(demo: Demo, root) -> str:
    return demo.repo if "://" in demo.repo else str(Path(root) / demo.repo)


def rollout_instance(policy_factory: PolicyFactory, demo: Demo, root=".", k: int = 4, max_steps: int = 40,
                     hint_level: int = 0, model_name: str = "unknown", *, backend: str = "auto",
                     grade_fn: GradeFn = grade, exclude_ids: set[str] | None = None,
                     sample_offset: int = 0, workers: int = 1, pool=None) -> list[Demo]:
    """k independent episodes in fresh RepoEnvs, each graded. Returns one Demo per episode, in
    sample order. `workers` > 1 (or a shared `pool`) runs the episodes concurrently."""
    _check_excluded([demo], exclude_ids)
    hint = build_hint(demo, hint_level)

    def one(i: int) -> Demo:
        traj: list[Step] = []
        patch, error, served = "", None, []
        g = GradeResult(False)
        try:
            policy = policy_factory()
            if hint_level > 0:
                policy = HintedPolicy(policy, hint, hint_level)
            with RepoEnv(_repo_path(demo, root), demo.base_commit, backend=backend) as env:
                res = run_agent(policy, env, demo.problem_statement, demo.test_cmd, max_steps)
            traj, patch = res.trajectory, res.patch
            served = sorted(getattr(policy, "served_models", None) or [])
            g = grade_fn(demo, patch, root)
        except Exception as e:  # one broken episode (API error, sandbox failure) must not kill the run
            error = f"{type(e).__name__}: {e}"
        fe = first_edit_index(traj)
        meta = {k_: demo.metadata[k_] for k_ in INSTANCE_META_KEYS if k_ in demo.metadata}
        meta.update({
            "source": f"teacher:{model_name}",
            "parent_id": demo.instance_id,
            "sample_idx": sample_offset + i,
            "resolved": bool(g.resolved),
            "grade": {"fail_to_pass": g.fail_to_pass, "pass_to_pass": g.pass_to_pass, "error": g.error},
            "hint_level": hint_level,
            "ran_tests_before_submit": ran_tests_before_submit(traj),
            "leakage_flags": leakage_flags(demo, traj, patch) if traj else [],
            "served_models": served,
            "weight": 1.0,
            "num_steps": len(traj),
            "steps_to_first_edit": fe,
            "patch_bucket": patch_bucket(patch),
            "submitted": bool(traj) and traj[-1].action.tool == "submit",
        })
        if error:
            meta["error"] = error
        return dataclasses.replace(demo, trajectory=traj, patch=patch, metadata=meta)

    return pmap(one, range(k), workers, pool)


def distill_instance(policy_factory: PolicyFactory, demo: Demo, root=".", k: int = 4, max_steps: int = 40,
                     hints: bool = True, model_name: str = "unknown", *, max_hint_level: int = MAX_HINT_LEVEL,
                     backend: str = "auto", grade_fn: GradeFn = grade,
                     exclude_ids: set[str] | None = None, workers: int = 1, pool=None) -> list[Demo]:
    """Unhinted k samples; if none resolved and hints enabled, escalate levels 1..max_hint_level
    (k samples each), stopping at the first level with >= 1 resolved. Returns ALL rollouts."""
    _check_excluded([demo], exclude_ids)
    rollouts: list[Demo] = []
    levels = [0] + (list(range(1, max_hint_level + 1)) if hints else [])
    for level in levels:
        batch = rollout_instance(policy_factory, demo, root, k, max_steps, level, model_name,
                                 backend=backend, grade_fn=grade_fn, sample_offset=len(rollouts),
                                 workers=workers, pool=pool)
        rollouts += batch
        if any(r.metadata["resolved"] for r in batch):
            break
    return rollouts


# ---- filtering ------------------------------------------------------------------------------------
def _first_edit_reference(rollouts: list[Demo]) -> dict[str, list[int]]:
    ref: dict[str, list[int]] = defaultdict(list)
    for r in rollouts:
        m = r.metadata
        if m["hint_level"] == 0 and m["resolved"] and m.get("steps_to_first_edit") is not None:
            ref[m["patch_bucket"]].append(m["steps_to_first_edit"])
            ref["*"].append(m["steps_to_first_edit"])
    return ref


def filter_rollouts(rollouts: list[Demo], max_per_instance: int = 2, *,
                    first_edit_reference: dict[str, list[int]] | None = None,
                    fast_edit_fraction: float = 0.5, downweight: float = 0.5,
                    min_bucket: int = 3) -> tuple[list[Demo], list[Demo], Counter]:
    """Apply §3b/§3c filters. Returns (kept, failed(unresolved), drop-reason counts).

    ``first_edit_reference`` maps patch bucket (and "*" = all) -> unhinted steps-to-first-edit
    values; defaults to the unhinted resolved rollouts in ``rollouts``."""
    ref = first_edit_reference if first_edit_reference is not None else _first_edit_reference(rollouts)
    dropped: Counter = Counter()
    failed: list[Demo] = []
    by_inst: dict[str, list[Demo]] = defaultdict(list)
    for r in rollouts:
        m = r.metadata
        if not m["resolved"]:
            failed.append(r)
            dropped["unresolved"] += 1
            continue
        if not m["ran_tests_before_submit"]:
            dropped["no_tests_before_submit"] += 1
            continue
        if m["hint_level"] > 0:
            leak = [f for f in DROP_REASONS if f in m["leakage_flags"]]
            if leak:
                dropped[leak[0]] += 1
                continue
            fe = m.get("steps_to_first_edit")
            vals = ref.get(m["patch_bucket"], [])
            if len(vals) < min_bucket:
                vals = ref.get("*", [])
            if fe is not None and len(vals) >= min_bucket and fe < fast_edit_fraction * statistics.median(vals):
                m["weight"] = downweight
                if "fast_first_edit" not in m["leakage_flags"]:
                    m["leakage_flags"].append("fast_first_edit")
        by_inst[m["parent_id"]].append(r)

    kept: list[Demo] = []
    for iid, cands in by_inst.items():
        cands.sort(key=lambda r: (len(r.trajectory), -r.metadata["weight"], r.metadata["sample_idx"]))
        sigs, patches, n = set(), set(), 0
        for r in cands:
            sig, np_ = action_signature(r.trajectory), normalize_patch(r.patch)
            if sig in sigs or np_ in patches:
                dropped["duplicate"] += 1
                continue
            if n >= max_per_instance:
                dropped["over_cap"] += 1
                continue
            sigs.add(sig)
            patches.add(np_)
            kept.append(r)
            n += 1
    return kept, failed, dropped


def distill(demos: list[Demo], policy_factory: PolicyFactory, root=".", k: int = 4, max_steps: int = 40,
            model_name: str = "unknown", *, exclude_ids: set[str] | None = None, hints: bool = True,
            max_per_instance: int = 2, max_hint_level: int = MAX_HINT_LEVEL, backend: str = "auto",
            grade_fn: GradeFn = grade, fast_edit_fraction: float = 0.5, downweight: float = 0.5,
            min_bucket: int = 3, on_rollout: Callable[[Demo], None] | None = None, workers: int = 1,
            ) -> tuple[list[Demo], list[Demo], dict]:
    """Run §3c (+§3b when ``hints``) over ``demos``. Returns (resolved_kept, failed, stats).

    Raises FrozenInstanceError before any rollout if a demo is in ``exclude_ids``.
    ``on_rollout`` is a cost hook called once per finished episode (from the calling thread).
    ``workers`` > 1 runs up to that many episodes at once across instances (results keep input
    order); size it to your API rate limit and sandbox capacity."""
    _check_excluded(demos, exclude_ids)
    all_rollouts: list[Demo] = []

    def per_instance(d: Demo, pool=None) -> list[Demo]:
        return distill_instance(policy_factory, d, root, k, max_steps, hints, model_name,
                                max_hint_level=max_hint_level, backend=backend, grade_fn=grade_fn, pool=pool)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers, thread_name_prefix="episode") as episodes:
            results = pmap(lambda d: per_instance(d, episodes), demos, workers)
    else:
        results = [per_instance(d) for d in demos]
    for rs in results:
        if on_rollout:
            for r in rs:
                on_rollout(r)
        all_rollouts += rs
    kept, failed, dropped = filter_rollouts(all_rollouts, max_per_instance, fast_edit_fraction=fast_edit_fraction,
                                            downweight=downweight, min_bucket=min_bucket)
    return kept, failed, _stats(demos, all_rollouts, kept, failed, dropped)


def _stats(demos, rollouts, kept, failed, dropped) -> dict:
    per_inst: dict[str, dict] = {d.instance_id: {"successes_by_level": {}, "attempts_by_level": {},
                                                 "resolved_level": None, "kept": 0} for d in demos}
    by_level: dict[int, dict] = {}
    steps_by_level: Counter = Counter()
    for r in rollouts:
        m = r.metadata
        lv, iid = m["hint_level"], m["parent_id"]
        p = per_inst.setdefault(iid, {"successes_by_level": {}, "attempts_by_level": {},
                                      "resolved_level": None, "kept": 0})
        p["attempts_by_level"][str(lv)] = p["attempts_by_level"].get(str(lv), 0) + 1
        p["successes_by_level"][str(lv)] = p["successes_by_level"].get(str(lv), 0) + int(m["resolved"])
        if m["resolved"] and (p["resolved_level"] is None or lv < p["resolved_level"]):
            p["resolved_level"] = lv
        b = by_level.setdefault(lv, {"attempted": 0, "resolved": 0, "instances": set()})
        b["attempted"] += 1
        b["resolved"] += int(m["resolved"])
        b["instances"].add(iid)
        steps_by_level[str(lv)] += len(r.trajectory)
    for r in kept:
        per_inst[r.metadata["parent_id"]]["kept"] += 1
    rate = {str(lv): {"attempted": b["attempted"], "resolved": b["resolved"], "instances": len(b["instances"]),
                      "rate": b["resolved"] / b["attempted"] if b["attempted"] else 0.0}
            for lv, b in sorted(by_level.items())}
    return {
        "instances": len(demos),
        "instances_with_kept": sum(1 for p in per_inst.values() if p["kept"]),
        "per_instance": per_inst,
        "resolve_rate_by_hint_level": rate,
        "rollouts": len(rollouts),
        "kept": len(kept),
        "failed": len(failed),
        "dropped": {r: dropped.get(r, 0) for r in DROP_REASONS},
        "flagged_unhinted": dict(Counter(f for r in rollouts if r.metadata["hint_level"] == 0
                                         and r.metadata["resolved"] for f in r.metadata["leakage_flags"])),
        "downweighted": sum(1 for r in kept if r.metadata["weight"] < 1.0),
        "errors": sum(1 for r in rollouts if "error" in r.metadata),
        "cost": {"episodes": len(rollouts), "steps": sum(len(r.trajectory) for r in rollouts),
                 "steps_by_level": dict(steps_by_level)},
    }


# ---- mixing -------------------------------------------------------------------------------------
def _inst(d: Demo) -> str:
    return d.metadata.get("parent_id") or d.instance_id


def _cap(items: list[Demo], cap: int | None) -> list[Demo]:
    if cap is None:
        return list(items)
    groups: dict[str, list[Demo]] = defaultdict(list)
    for d in items:
        groups[_inst(d)].append(d)
    out = []
    for g in groups.values():
        g = sorted(g, key=lambda d: (-float(d.metadata.get("weight", 1.0)), len(d.trajectory)))
        out += g[:cap]
    return out


def _spread_sample(items: list[Demo], n: int, rng: random.Random) -> list[Demo]:
    """Round-robin over shuffled instances so a reduced sample covers as many instances as possible."""
    if n >= len(items):
        return list(items)
    groups: dict[str, list[Demo]] = defaultdict(list)
    for d in items:
        groups[_inst(d)].append(d)
    order = list(groups.values())
    rng.shuffle(order)
    for g in order:
        rng.shuffle(g)
    out: list[Demo] = []
    depth = 0
    while len(out) < n:
        for g in order:
            if depth < len(g) and len(out) < n:
                out.append(g[depth])
        depth += 1
    return out


def mix_datasets(teacher: list[Demo], engineer: list[Demo], ratio: tuple[int, int] = (3, 1), seed: int = 0,
                 cap_per_instance: int | None = None) -> list[Demo]:
    """Mix teacher:engineer by trajectory count (as close to ``ratio`` as the smaller side allows).

    ``cap_per_instance`` caps records per instance within each source before mixing (highest weight,
    then shortest kept). Each output record gets ``metadata.mix_source`` in {"teacher","engineer"}."""
    a, b = ratio
    if a < 0 or b < 0 or a + b == 0:
        raise ValueError(f"bad ratio {ratio}")
    rng = random.Random(seed)
    t, e = _cap(teacher, cap_per_instance), _cap(engineer, cap_per_instance)
    if b == 0:
        nt, ne = len(t), 0
    elif a == 0:
        nt, ne = 0, len(e)
    else:
        unit = min(len(t) / a, len(e) / b)
        nt, ne = int(round(unit * a)), int(round(unit * b))
    picked = [(d, "teacher") for d in _spread_sample(t, nt, rng)] + \
             [(d, "engineer") for d in _spread_sample(e, ne, rng)]
    rng.shuffle(picked)
    return [dataclasses.replace(d, metadata={"source": "engineer", **d.metadata, "mix_source": src})
            if src == "engineer" else dataclasses.replace(d, metadata={**d.metadata, "mix_source": src})
            for d, src in picked]


# ---- CLI ----------------------------------------------------------------------------------------
def load_exclude_ids(path: str | Path) -> set[str]:
    """JSON manifest ({"instances": {id: ...}} / {"instances": [id | {"instance_id": id}]} / [ids])
    or newline-separated ids (blank lines and # comments ignored)."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    if isinstance(data, dict):
        data = data.get("instances", data)
    if isinstance(data, dict):
        return set(map(str, data))
    if isinstance(data, list):
        return {str(x["instance_id"]) if isinstance(x, dict) else str(x) for x in data}
    raise ValueError(f"unrecognized exclude-ids manifest: {path}")


def _parse_ratio(s: str) -> tuple[int, int]:
    try:
        a, b = s.split(":")
        return int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ratio must look like 3:1, got {s!r}") from None


def cmd_distill(a) -> int:
    demos = load_demos(a.demos)
    exclude = load_exclude_ids(a.exclude_ids) if a.exclude_ids else set()
    factory = lambda: policies.make_policy(a.model)  # noqa: E731
    try:
        kept, failed, stats = distill(demos, factory, a.root, a.k, a.max_steps, a.model, exclude_ids=exclude,
                                      hints=a.hints, max_per_instance=a.max_per_instance, backend=a.backend,
                                      workers=a.workers)
    except FrozenInstanceError as e:
        print(f"error: {e}")
        return 2
    save_demos(kept, a.out_resolved)
    save_demos(failed, a.out_failed)
    if a.stats:
        Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(a.stats).write_text(json.dumps(stats, indent=2))
    print(f"kept {len(kept)} resolved, {len(failed)} failed rollouts from {len(demos)} instances "
          f"({stats['cost']['episodes']} episodes, {stats['cost']['steps']} steps)")
    return 0


def cmd_mix(a) -> int:
    mixed = mix_datasets(load_demos(a.teacher), load_demos(a.engineer), a.ratio, a.seed, a.cap_per_instance)
    save_demos(mixed, a.out)
    c = Counter(d.metadata["mix_source"] for d in mixed)
    print(f"wrote {len(mixed)} records (teacher {c['teacher']}, engineer {c['engineer']}) to {a.out}")
    return 0


def register(subparsers) -> None:
    d = subparsers.add_parser("distill", help="§3b/§3c rejection-sampled (optionally hinted) teacher rollouts")
    d.add_argument("demos", help="input demos JSONL (train split only)")
    d.add_argument("--model", required=True)
    d.add_argument("--k", type=int, default=4)
    d.add_argument("--max-steps", type=int, default=40)
    d.add_argument("--out-resolved", default="runs/distill/resolved.jsonl")
    d.add_argument("--out-failed", default="runs/distill/failed.jsonl")
    d.add_argument("--stats", default=None)
    d.add_argument("--hints", action="store_true", help="escalate the §3b hint ladder when unhinted fails")
    d.add_argument("--max-per-instance", type=int, default=2)
    d.add_argument("--exclude-ids", default=None, help="frozen-test manifest (JSON) or newline-separated ids")
    d.add_argument("--root", default=".")
    d.add_argument("--backend", default="auto", choices=["auto", "docker", "local"])
    d.add_argument("--workers", type=int, default=1, help="concurrent episodes (threads)")
    d.set_defaults(fn=cmd_distill)

    m = subparsers.add_parser("mix", help="mix teacher and engineer demos at a trajectory-count ratio")
    m.add_argument("--teacher", required=True)
    m.add_argument("--engineer", required=True)
    m.add_argument("--ratio", type=_parse_ratio, default=(3, 1))
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--cap-per-instance", type=int, default=None)
    m.add_argument("--out", default="runs/mix/mixed.jsonl")
    m.set_defaults(fn=cmd_mix)
