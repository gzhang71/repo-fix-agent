"""§3f Verifier / reward model + best-of-n reranking.

* `extract_features(record)`  - cheap, interpretable features from a rollout / demo record.
* `tag_hard_negatives`        - unresolved rollouts whose own last test run passed.
* `FeatureVerifier`           - stdlib logistic regression (standardized features, L2, batch GD).
* `JudgeVerifier(llm)`        - generative judge (YES/NO) over issue + final patch + last test output.
* `best_of_n` / `run_best_of_n` / `evaluate_best_of_n` - inference-time reranking and its eval.

A "scorer" is any `Callable[[Demo], float]` returning P(resolved); both verifiers qualify.

CLI:
  repo-fix verifier-train --rollouts R.jsonl [R2.jsonl ...] [--demos D.jsonl] --out model.json [--exclude-ids FILE]
  repo-fix best-of-n --repo PATH --issue-file F --model M --n 4 --verifier model.json [--apply]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..schema import Demo, Step, load_demos

Scorer = Callable[[Demo], float]

DEFAULT_TEST_INFRA_PATTERNS = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                               ".github/", "noxfile.py")
EDIT_TOOLS = {"str_replace", "create_file", "insert", "undo_edit"}

_EXIT_RE = re.compile(r"\[exit code (-?\d+)\]")
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERRORS_RE = re.compile(r"(\d+) errors?\b")
_BASH_EDIT_RE = re.compile(r"\bsed\s+-i\b|(?<![<>0-9&])>(?!&)\s*[\w./-]+\.py\b|\btee\s+[\w./-]+\.py\b")
_REPRO_RE = re.compile(r"\bpython[0-9.]*\s+(-c\b|(?!-m\b)[\w./-]+\.py\b)")


# ---- frozen-test guard (shared by rl.py / retrieval.py) ------------------------------------
class ExcludedInstanceError(ValueError):
    """Raised when a frozen dev/test instance would be used for training or indexing."""


def load_exclude_ids(path: str | Path | None) -> set[str]:
    """Read a JSON manifest ({"instances": {id: ...}} / {"instances": [...]} / [...]) or a
    newline-separated id list."""
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
    if isinstance(data, list):
        return {str(x["instance_id"]) if isinstance(x, dict) else str(x) for x in data}
    raise ValueError(f"unrecognized exclude-ids manifest: {path}")


def check_not_excluded(records: Iterable[Demo], exclude_ids: set[str] | None, purpose: str = "training") -> None:
    if not exclude_ids:
        return
    bad = sorted({r.instance_id for r in records
                  if r.instance_id in exclude_ids or str((r.metadata or {}).get("parent_id")) in exclude_ids})
    if bad:
        raise ExcludedInstanceError(
            f"{len(bad)} frozen (excluded) instance(s) would be used for {purpose}: {bad[:10]}")


# ---- record helpers -------------------------------------------------------------------------
def record_patch(rec: Demo) -> str:
    """The candidate patch of a rollout (sibling modules may store it in metadata); else `patch`."""
    md = rec.metadata or {}
    for k in ("model_patch", "candidate_patch", "final_patch"):
        if isinstance(md.get(k), str):
            return md[k]
    return rec.patch or ""


def group_key(rec: Demo) -> str:
    return str((rec.metadata or {}).get("parent_id") or rec.instance_id)


def label_of(rec: Demo) -> int | None:
    """1 = resolved rollout or engineer demo, 0 = unresolved rollout, None = unknown."""
    md = rec.metadata or {}
    if isinstance(md.get("resolved"), bool):
        return int(md["resolved"])
    if str(md.get("source", "")).startswith("engineer"):
        return 1
    return None


def parse_patch(patch: str) -> dict[str, Any]:
    files: list[str] = []
    hunks = added = removed = 0
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if m and m.group(2) not in files:
                files.append(m.group(2))
        elif line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p != "/dev/null":
                p = p[2:] if p[:2] in ("a/", "b/") else p
                if p not in files:
                    files.append(p)
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"files": files, "hunks": hunks, "added": added, "removed": removed}


def touches_test_infra(files: Iterable[str], patterns: Sequence[str] = DEFAULT_TEST_INFRA_PATTERNS) -> list[str]:
    hits = []
    for f in files:
        base = f.rsplit("/", 1)[-1]
        for p in patterns:
            if (p.endswith("/") and (f.startswith(p) or f"/{p}" in f"/{f}")) or base == p:
                hits.append(f)
                break
    return hits


def is_test_step(step: Step) -> bool:
    if step.action.tool == "run_tests":
        return True
    if step.action.tool == "bash":
        cmd = str(step.action.args.get("command", ""))
        return bool(re.search(r"\bpytest\b|\brun_tests\b|\bunittest\b|\btox\b", cmd))
    return False


def is_edit_step(step: Step) -> bool:
    if step.action.tool in EDIT_TOOLS:
        return True
    if step.action.tool == "bash":
        return bool(_BASH_EDIT_RE.search(str(step.action.args.get("command", ""))))
    return False


def parse_exit_code(obs: str) -> int | None:
    m = _EXIT_RE.findall(obs or "")
    return int(m[-1]) if m else None


def parse_pytest_counts(obs: str) -> tuple[int, int]:
    """(passed, failed+errors) from the last pytest summary in `obs`."""
    def last(rx):
        m = rx.findall(obs or "")
        return int(m[-1]) if m else 0
    return last(_PASSED_RE), last(_FAILED_RE) + last(_ERRORS_RE)


def last_test_step(traj: Sequence[Step]) -> Step | None:
    for st in reversed(traj):
        if is_test_step(st):
            return st
    return None


def own_tests_passed(rec: Demo) -> bool:
    st = last_test_step(rec.trajectory)
    if st is None:
        return False
    rc = parse_exit_code(st.observation)
    passed, failed = parse_pytest_counts(st.observation)
    return rc == 0 if rc is not None else (passed > 0 and failed == 0)


FEATURE_NAMES = [
    "ran_tests_before_submit", "last_test_exit_zero", "last_test_exit_nonzero", "last_test_passed",
    "last_test_failed", "patch_lines", "patch_files", "patch_hunks", "edited_test_infra", "empty_patch",
    "num_steps", "num_tool_errors", "reproduced_issue", "submitted",
]


def extract_features(rec: Demo, test_infra_patterns: Sequence[str] = DEFAULT_TEST_INFRA_PATTERNS) -> dict[str, float]:
    traj = rec.trajectory
    submit_idx = next((i for i, s in enumerate(traj) if s.action.tool == "submit"), len(traj))
    ran_tests = any(is_test_step(s) for s in traj[:submit_idx])
    lt = last_test_step(traj)
    rc = parse_exit_code(lt.observation) if lt else None
    passed, failed = parse_pytest_counts(lt.observation) if lt else (0, 0)
    pp = parse_patch(record_patch(rec))
    first_edit = next((i for i, s in enumerate(traj) if is_edit_step(s)), len(traj))
    reproduced = any(s.action.tool == "bash" and not is_test_step(s)
                     and _REPRO_RE.search(str(s.action.args.get("command", ""))) for s in traj[:first_edit])
    md = rec.metadata or {}
    submitted = md.get("submitted")
    if not isinstance(submitted, bool):
        submitted = submit_idx < len(traj)
    return {
        "ran_tests_before_submit": float(ran_tests),
        "last_test_exit_zero": float(rc == 0),
        "last_test_exit_nonzero": float(rc is not None and rc != 0),
        "last_test_passed": float(passed),
        "last_test_failed": float(failed),
        "patch_lines": float(pp["added"] + pp["removed"]),
        "patch_files": float(len(pp["files"])),
        "patch_hunks": float(pp["hunks"]),
        "edited_test_infra": float(bool(touches_test_infra(pp["files"], test_infra_patterns))),
        "empty_patch": float(not record_patch(rec).strip()),
        "num_steps": float(len(traj)),
        "num_tool_errors": float(sum(s.observation.startswith("Error:") for s in traj)),
        "reproduced_issue": float(bool(reproduced)),
        "submitted": float(bool(submitted)),
    }


def tag_hard_negatives(records: Iterable[Demo]) -> int:
    """Set metadata.hard_negative on every record that has a `resolved` flag; returns the count."""
    n = 0
    for r in records:
        md = r.metadata if r.metadata is not None else {}
        if md.get("resolved") is False:
            md["hard_negative"] = own_tests_passed(r)
            n += md["hard_negative"]
        elif md.get("resolved") is True:
            md["hard_negative"] = False
        r.metadata = md
    return n


# ---- metrics --------------------------------------------------------------------------------
def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based (Mann-Whitney U) AUROC with average ranks for ties; NaN if one class is missing."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---- logistic-regression verifier -----------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class FeatureVerifier:
    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES):
        self.feature_names = list(feature_names)
        self.weights = [0.0] * len(self.feature_names)
        self.bias = 0.0
        self.mean = [0.0] * len(self.feature_names)
        self.std = [1.0] * len(self.feature_names)
        self.meta: dict[str, Any] = {}

    # vectors
    def vectorize(self, feats: Mapping[str, float]) -> list[float]:
        return [float(feats.get(k, 0.0)) for k in self.feature_names]

    def _std(self, x: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mean, self.std)]

    def fit_arrays(self, X: Sequence[Sequence[float]], y: Sequence[int], epochs: int = 500,
                   lr: float = 0.5, l2: float = 1e-3) -> "FeatureVerifier":
        if not X:
            raise ValueError("no training examples")
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        self.std = [math.sqrt(sum((r[j] - self.mean[j]) ** 2 for r in X) / n) or 1.0 for j in range(d)]
        Z = [self._std(r) for r in X]
        w, b = [0.0] * d, 0.0
        for _ in range(epochs):
            gw, gb = [0.0] * d, 0.0
            for z, t in zip(Z, y):
                err = _sigmoid(sum(wi * zi for wi, zi in zip(w, z)) + b) - t
                gb += err
                for j in range(d):
                    gw[j] += err * z[j]
            w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, gw)]
            b -= lr * gb / n
        self.weights, self.bias = w, b
        return self

    def fit(self, records: Iterable[Demo], **kw) -> "FeatureVerifier":
        X, y = [], []
        for r in records:
            lab = label_of(r)
            if lab is not None:
                X.append(self.vectorize(extract_features(r)))
                y.append(lab)
        return self.fit_arrays(X, y, **kw)

    def predict_proba_features(self, x: Sequence[float] | Mapping[str, float]) -> float:
        v = self.vectorize(x) if isinstance(x, Mapping) else list(x)
        return _sigmoid(sum(w * z for w, z in zip(self.weights, self._std(v))) + self.bias)

    def predict_proba(self, rec: Demo) -> float:
        return self.predict_proba_features(extract_features(rec))

    __call__ = predict_proba

    def to_dict(self) -> dict[str, Any]:
        return {"type": "feature_logreg", "feature_names": self.feature_names, "weights": self.weights,
                "bias": self.bias, "mean": self.mean, "std": self.std, "meta": self.meta}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureVerifier":
        d = json.loads(Path(path).read_text())
        v = cls(d["feature_names"])
        v.weights, v.bias, v.mean, v.std = d["weights"], d["bias"], d["mean"], d["std"]
        v.meta = d.get("meta", {})
        return v


# ---- generative judge -----------------------------------------------------------------------
JUDGE_SYSTEM = ("You are a meticulous code reviewer. Decide whether a candidate patch fully resolves a "
                "GitHub-style issue without breaking existing behavior. Hidden tests will be run; passing "
                "the agent's own tests is not sufficient evidence by itself.")

JUDGE_TEMPLATE = """<issue>
{issue}
</issue>

<candidate_patch>
{patch}
</candidate_patch>

<last_test_output>
{tests}
</last_test_output>

Does the patch correctly resolve the issue? End your answer with exactly two lines:
VERDICT: YES or VERDICT: NO
CONFIDENCE: a number between 0 and 1 (your confidence in the verdict)"""


def parse_judge_answer(text: str, default_conf: float = 0.85) -> float | None:
    """Map a judge reply to P(resolved); None if no verdict can be found."""
    t = text or ""
    m = re.findall(r"VERDICT\s*[:=]\s*\**\s*(YES|NO)\b", t, re.I)
    verdict = m[-1].upper() if m else None
    if verdict is None:
        words = re.findall(r"\b(YES|NO)\b", t, re.I)
        verdict = words[-1].upper() if words else None
    if verdict is None:
        return None
    conf = default_conf
    c = re.findall(r"CONFIDENCE\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", t, re.I)
    if c:
        val = float(c[-1][0])
        val = val / 100 if (c[-1][1] or val > 1) else val
        conf = min(1.0, max(0.0, val))
    return conf if verdict == "YES" else 1.0 - conf


class JudgeVerifier:
    def __init__(self, llm: Callable[[str, str], str], max_patch_chars: int = 6000, max_test_chars: int = 3000):
        self.llm, self.max_patch_chars, self.max_test_chars = llm, max_patch_chars, max_test_chars

    def prompt(self, rec: Demo) -> str:
        lt = last_test_step(rec.trajectory)
        tests = lt.observation[-self.max_test_chars:] if lt else "(the agent did not run tests)"
        patch = record_patch(rec)[: self.max_patch_chars] or "(empty patch)"
        return JUDGE_TEMPLATE.format(issue=rec.problem_statement.strip(), patch=patch, tests=tests)

    def __call__(self, rec: Demo) -> float:
        if not record_patch(rec).strip():
            return 0.0
        try:
            p = parse_judge_answer(self.llm(JUDGE_SYSTEM, self.prompt(rec)))
        except Exception:  # noqa: BLE001 - a failed judge call must not kill reranking
            return 0.5
        return 0.5 if p is None else p

    predict_proba = __call__


# ---- best-of-n ------------------------------------------------------------------------------
def normalize_patch(patch: str) -> str:
    """Canonical form for agreement voting: file headers + changed lines, whitespace-insensitive."""
    out = []
    for line in (patch or "").splitlines():
        if line.startswith("+++ "):
            out.append("F " + line[4:].strip().removeprefix("b/"))
        elif line.startswith("--- ") or line.startswith("diff --git") or line.startswith("index "):
            continue
        elif line[:1] in "+-" and line[1:].strip():
            out.append(line[0] + " ".join(line[1:].split()))
    return "\n".join(out)


@dataclass
class RankedCandidate:
    index: int
    record: Demo
    patch: str
    verifier_score: float
    agreement: float
    combined: float
    own_tests_passed: bool
    explanation: str = ""


def best_of_n(candidates: Sequence[Demo], scorer: Scorer, vote_weight: float = 0.3) -> list[RankedCandidate]:
    """Rank candidates by (1-w)*verifier + w*agreement, tie-break on own tests passed, then order."""
    n = len(candidates)
    keys = [normalize_patch(record_patch(c)) for c in candidates]
    counts: dict[str, int] = {}
    for k in keys:
        if k:
            counts[k] = counts.get(k, 0) + 1
    ranked = []
    for i, c in enumerate(candidates):
        score = float(scorer(c))
        agree = counts.get(keys[i], 0) / n if keys[i] else 0.0
        own = own_tests_passed(c)
        comb = (1 - vote_weight) * score + vote_weight * agree
        expl = (f"verifier={score:.3f}, agreement={counts.get(keys[i], 0)}/{n}"
                f"{' (empty patch)' if not keys[i] else ''}, own_tests_passed={own}, "
                f"combined={comb:.3f}")
        ranked.append(RankedCandidate(i, c, record_patch(c), score, agree, comb, own, expl))
    ranked.sort(key=lambda r: (-round(r.combined, 9), not r.own_tests_passed, r.index))
    for pos, r in enumerate(ranked):
        r.explanation = f"rank {pos + 1}: " + r.explanation
    return ranked


@dataclass
class BestOfNResult:
    patch: str
    ranked: list[RankedCandidate]
    candidates: list[Demo] = field(default_factory=list)


def run_best_of_n(policy_factory: Callable[[], Any], problem_statement: str, repo: str | Path, n: int,
                  scorer: Scorer, test_cmd: str | None = None, max_steps: int = 40,
                  base_commit: str | None = None, backend: str = "auto",
                  vote_weight: float = 0.3, workers: int = 1) -> BestOfNResult:
    """Inference path: sample n trajectories in fresh sandboxes (concurrently with `workers` > 1),
    rerank, return the chosen patch."""
    from ..agent import run_agent
    from ..env import RepoEnv
    from .parallel import pmap

    def one(i: int) -> Demo:
        with RepoEnv(repo, base_commit, backend=backend) as env:
            res = run_agent(policy_factory(), env, problem_statement, test_cmd, max_steps)
        return Demo(instance_id=f"best-of-n-{i}", repo=str(repo), base_commit=base_commit,
                    problem_statement=problem_statement, test_cmd=test_cmd or "", test_patch="",
                    FAIL_TO_PASS=[], PASS_TO_PASS=[], patch=res.patch, trajectory=res.trajectory,
                    metadata={"source": "best_of_n", "sample_idx": i, "submitted": res.submitted})

    cands = pmap(one, range(n), workers)
    ranked = best_of_n(cands, scorer, vote_weight)
    return BestOfNResult(ranked[0].patch if ranked else "", ranked, cands)


def group_by_parent(records: Iterable[Demo]) -> dict[str, list[Demo]]:
    groups: dict[str, list[Demo]] = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    return groups


def evaluate_best_of_n(demo_groups: Mapping[str, Sequence[Demo]] | Iterable[Sequence[Demo]], scorer: Scorer,
                       vote_weight: float = 0.3) -> dict[str, float]:
    """pass@1 (mean resolve rate), best-of-n via scorer, oracle pass@n, and fraction of gap closed."""
    groups = list(demo_groups.values()) if isinstance(demo_groups, Mapping) else list(demo_groups)
    groups = [[r for r in g if isinstance((r.metadata or {}).get("resolved"), bool)] for g in groups]
    groups = [g for g in groups if g]
    if not groups:
        return {"n_instances": 0, "pass@1": float("nan"), "best_of_n": float("nan"), "oracle_pass@n": float("nan"),
                "gap_closed": float("nan")}
    p1 = sum(sum(r.metadata["resolved"] for r in g) / len(g) for g in groups) / len(groups)
    bon = sum(bool(best_of_n(g, scorer, vote_weight)[0].record.metadata["resolved"]) for g in groups) / len(groups)
    oracle = sum(any(r.metadata["resolved"] for r in g) for g in groups) / len(groups)
    gap = (bon - p1) / (oracle - p1) if oracle > p1 else float("nan")
    return {"n_instances": len(groups), "pass@1": p1, "best_of_n": bon, "oracle_pass@n": oracle, "gap_closed": gap}


# ---- train / dev split ----------------------------------------------------------------------
def split_by_parent(records: Sequence[Demo], dev_frac: float = 0.2, seed: str = "verifier") -> tuple[list[Demo], list[Demo]]:
    keys = sorted({group_key(r) for r in records},
                  key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    if len(keys) < 2:
        return list(records), []
    n_dev = min(len(keys) - 1, max(1, round(dev_frac * len(keys))))
    dev_keys = set(keys[:n_dev])
    return ([r for r in records if group_key(r) not in dev_keys], [r for r in records if group_key(r) in dev_keys])


def load_scorer(spec: str) -> Scorer:
    """`model.json` -> FeatureVerifier; `judge:<model>` -> JudgeVerifier(make_text_llm(model))."""
    if spec.startswith("judge:"):
        from ..llm import make_text_llm
        return JudgeVerifier(make_text_llm(spec.split(":", 1)[1]))
    return FeatureVerifier.load(spec)


# ---- CLI ------------------------------------------------------------------------------------
def cmd_verifier_train(a) -> int:
    records: list[Demo] = []
    for p in a.rollouts:
        records += load_demos(p)
    for p in a.demos or []:
        for d in load_demos(p):
            d.metadata = {**(d.metadata or {})}
            d.metadata.setdefault("source", "engineer")
            records.append(d)
    try:
        check_not_excluded(records, load_exclude_ids(a.exclude_ids), "verifier training")
    except ExcludedInstanceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    n_hard = tag_hard_negatives(records)
    labeled = [r for r in records if label_of(r) is not None]
    n_pos = sum(label_of(r) for r in labeled)
    print(f"{len(labeled)} labeled records ({n_pos} pos / {len(labeled) - n_pos} neg, {n_hard} hard negatives)")
    if not labeled or n_pos in (0, len(labeled)):
        print("error: need both positive and negative examples", file=sys.stderr)
        return 1
    train, dev = split_by_parent(labeled, a.dev_frac)
    dev_auc = float("nan")
    if dev and len({label_of(r) for r in train}) == 2:
        v = FeatureVerifier().fit(train, epochs=a.epochs, l2=a.l2)
        dev_auc = auroc([v(r) for r in dev], [label_of(r) for r in dev])
        print(f"dev AUROC = {dev_auc:.3f}  (train {len(train)} / dev {len(dev)} records, split by parent instance)")
        hn = [r for r in dev if r.metadata.get("hard_negative")]
        if hn:
            print(f"dev hard negatives: mean score {sum(v(r) for r in hn) / len(hn):.3f} (n={len(hn)})")
        rollout_groups = {k: g for k, g in group_by_parent(
            [r for r in dev if isinstance(r.metadata.get("resolved"), bool)]).items() if len(g) > 1}
        if rollout_groups:
            print("dev best-of-n:", json.dumps(evaluate_best_of_n(rollout_groups, v)))
    else:
        print("dev AUROC = n/a (not enough parent instances / classes for a holdout split)")
    model = FeatureVerifier().fit(labeled, epochs=a.epochs, l2=a.l2)
    model.meta = {"dev_auroc": None if math.isnan(dev_auc) else dev_auc, "n_train": len(labeled),
                  "n_pos": n_pos, "n_hard_negatives": n_hard}
    model.save(a.out)
    print(f"wrote {a.out}")
    return 0


def cmd_best_of_n(a) -> int:
    from .. import policies

    issue = Path(a.issue_file).read_text() if a.issue_file else a.issue
    scorer = load_scorer(a.verifier)
    res = run_best_of_n(lambda: policies.make_policy(a.model), issue, a.repo, a.n, scorer,
                        test_cmd=a.test_cmd, max_steps=a.max_steps, backend=a.backend, vote_weight=a.vote_weight,
                        workers=a.workers)
    for r in res.ranked:
        print(r.explanation, file=sys.stderr)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"patch": res.patch, "ranking": [
            {"sample_idx": r.index, "verifier": r.verifier_score, "agreement": r.agreement,
             "combined": r.combined, "own_tests_passed": r.own_tests_passed} for r in res.ranked]}, indent=2))
    print(res.patch or "(no changes)")
    if a.apply and res.patch:
        subprocess.run(["git", "apply", "-"], cwd=a.repo, input=res.patch, text=True, check=True)
        print(f"Applied to {a.repo}", file=sys.stderr)
    return 0 if res.patch else 1


def register(subparsers) -> None:
    t = subparsers.add_parser("verifier-train", help="§3f train the feature verifier on graded rollouts")
    t.add_argument("--rollouts", nargs="+", required=True)
    t.add_argument("--demos", nargs="*", default=None, help="engineer demos (positives)")
    t.add_argument("--out", default="runs/verifier/model.json")
    t.add_argument("--exclude-ids", default=None)
    t.add_argument("--dev-frac", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--l2", type=float, default=1e-3)
    t.set_defaults(fn=cmd_verifier_train)

    b = subparsers.add_parser("best-of-n", help="§3f sample n trajectories and rerank with a verifier")
    b.add_argument("--repo", required=True)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--issue")
    g.add_argument("--issue-file")
    b.add_argument("--model", default="student")
    b.add_argument("--n", type=int, default=4)
    b.add_argument("--workers", type=int, default=1, help="sample the n candidates concurrently")
    b.add_argument("--verifier", required=True, help="model.json from verifier-train, or judge:<model>")
    b.add_argument("--vote-weight", type=float, default=0.3)
    b.add_argument("--test-cmd", default=None)
    b.add_argument("--max-steps", type=int, default=40)
    b.add_argument("--backend", default="auto")
    b.add_argument("--out", default=None, help="optional JSON report path")
    b.add_argument("--apply", action="store_true")
    b.set_defaults(fn=cmd_best_of_n)
