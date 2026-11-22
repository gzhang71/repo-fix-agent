"""§2.1 audit: validate x N (flakiness), trajectory replay, exact + near dedup, patch size,
trajectory length, test leakage, secrets/PII, optional LLM bug-type labels and issue-sufficiency.

Every check is a plain function over a `Demo` so other pipeline steps can reuse them; `audit_demos`
chains them in the order of the proposal and returns the kept demos plus an `AuditReport`.

Filters (drop):   invalid -> flaky -> exact duplicate
Flags (keep):     replay failure, leakage, secrets, length, near-dup clusters (metadata.cluster_id)

Metadata written: replay_ok, flaky, cluster_id, patch_size, leakage, secrets, public, bug_type,
issue_sufficient, repo_group, length_flags.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import posixpath
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .. import grader
from ..schema import Demo, load_demos, save_demos

log = logging.getLogger(__name__)

MAX_STEPS = 40
MAX_TOKENS = 64_000
CHARS_PER_TOKEN = 3.5
REPO_DROP_WARN = 0.20
REPO_SHARE_WARN = 0.10


# ============================================================================================
# helpers
# ============================================================================================
def _repo_path(demo: Demo, root: str | Path) -> str:
    return demo.repo if "://" in demo.repo else str(Path(root) / demo.repo)


def repo_group(demo: Demo) -> str:
    return demo.metadata.get("repo_group") or demo.repo


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def normalize_patch(patch: str) -> str:
    """Patch with headers that don't affect content (index lines, hunk headers' line numbers and
    function context) and all intra-line whitespace removed; blank +/- lines dropped."""
    out = []
    for line in (patch or "").splitlines():
        if line.startswith(("index ", "similarity index", "\\ No newline")):
            continue
        if line.startswith("@@"):
            out.append("@@")
            continue
        if line.startswith(("+++ ", "--- ", "diff --git")):
            out.append(re.sub(r"\s+", " ", line).strip())
            continue
        if line[:1] in "+- " and line:
            body = re.sub(r"\s+", "", line[1:])
            if body or line[0] == " ":
                out.append(line[0] + body)
    return "\n".join(out).strip()


def _patch_changes(patch: str) -> dict[str, list[str]]:
    """file -> ordered list of whitespace-free +/- lines (context and blank changes ignored)."""
    changes: dict[str, list[str]] = defaultdict(list)
    cur = None
    for line in (patch or "").splitlines():
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            cur = m.group(2) if m else line
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                cur = p[2:] if p.startswith("b/") else p
            continue
        if line.startswith("--- "):
            continue
        if line[:1] in "+-" and cur is not None:
            body = re.sub(r"\s+", "", line[1:])
            if body:
                changes[cur].append(line[0] + body)
    return {k: v for k, v in changes.items() if v}


def patches_equivalent(a: str, b: str) -> bool:
    """Same changed lines per file, ignoring whitespace, context, hunk headers and line numbers."""
    return _patch_changes(a) == _patch_changes(b)


def patch_files(patch: str, created_only: bool = False) -> list[str]:
    files: list[str] = []
    created: set[str] = set()
    cur = None
    for line in (patch or "").splitlines():
        m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
        if m:
            cur = m.group(2)
            if cur not in files:
                files.append(cur)
            continue
        if line.startswith("new file mode") and cur:
            created.add(cur)
        elif line.startswith("--- /dev/null"):
            created.add("__pending__")
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                p = p[2:] if p.startswith("b/") else p
                if p not in files:
                    files.append(p)
                if "__pending__" in created:
                    created.discard("__pending__")
                    created.add(p)
                cur = p
    return [f for f in files if f in created] if created_only else files


# ============================================================================================
# validation x N / flakiness
# ============================================================================================
GradeFn = Callable[..., "grader.GradeResult"]


def _signature(r) -> tuple:
    return (tuple(sorted(r.fail_to_pass.items())), tuple(sorted(r.pass_to_pass.items())),
            bool(r.error), bool(r.resolved))


def validate_n(demo: Demo, root: str | Path = ".", runs: int = 3,
               grade_fn: GradeFn | None = None) -> tuple[bool, bool, list[str]]:
    """Run `grader.validate_instance`'s checks `runs` times.

    Returns (valid, flaky, problems): valid = no problems in any run; flaky = any F2P/P2P result
    (before or after the gold patch) differs between runs."""
    grade_fn = grade_fn or grader.grade  # looked up at call time so tests can monkeypatch
    problems: list[str] = []
    sigs: set[tuple] = set()
    for i in range(max(1, runs)):
        try:
            before = grade_fn(demo, "", root)
            after = grade_fn(demo, demo.patch, root)
        except Exception as e:  # env build failure etc. counts as invalid, not as a crash
            problems.append(f"run {i + 1}: grading raised {type(e).__name__}: {e}")
            sigs.add(("error", type(e).__name__))
            continue
        sigs.add((_signature(before), _signature(after)))
        run_problems = []
        if any(before.fail_to_pass.values()):
            run_problems.append(f"FAIL_TO_PASS already passing without fix: {before.fail_to_pass}")
        if before.error:
            run_problems.append(f"baseline grading error: {before.error}")
        if not all(before.pass_to_pass.values()):
            run_problems.append(f"PASS_TO_PASS failing before fix: {before.pass_to_pass}")
        if not after.resolved:
            run_problems.append(f"gold patch does not resolve: {after}")
        for p in run_problems:
            msg = p if runs <= 1 else f"run {i + 1}: {p}"
            problems.append(msg)
    return (not problems, len(sigs) > 1, problems)


# ============================================================================================
# trajectory replay
# ============================================================================================
@dataclass
class ReplayResult:
    ok: bool
    diff: str = ""
    observations: list[str] = field(default_factory=list)
    error: str | None = None


def replay(demo: Demo, root: str | Path = ".", backend: str = "auto") -> ReplayResult:
    """Re-execute `demo.trajectory` actions in a fresh RepoEnv; ok iff the final diff equals
    `demo.patch` modulo whitespace (test files are already excluded by `env.diff`)."""
    from ..env import RepoEnv

    try:
        with RepoEnv(_repo_path(demo, root), demo.base_commit, backend=backend) as env:
            obs = [env.execute(s.action) for s in demo.trajectory]
            diff = env.diff()
    except Exception as e:
        return ReplayResult(False, error=f"{type(e).__name__}: {e}")
    return ReplayResult(patches_equivalent(diff, demo.patch), diff, obs)


def replay_check(demo: Demo, root: str | Path = ".", backend: str = "auto") -> bool:
    return replay(demo, root, backend).ok


# ============================================================================================
# dedup / near-dup
# ============================================================================================
def exact_dedup(demos: list[Demo]) -> tuple[list[Demo], list[tuple[str, str, str]]]:
    """Keep the first demo per normalized-patch hash and per normalized-issue hash.
    Returns (kept, dropped) where dropped = [(instance_id, duplicate_of, which_hash)]."""
    seen_patch: dict[str, str] = {}
    seen_issue: dict[str, str] = {}
    kept, dropped = [], []
    for d in demos:
        np_, ni = normalize_patch(d.patch), normalize_text(d.problem_statement)
        hp, hi = (_sha(np_) if np_ else None), (_sha(ni) if ni else None)
        if hp and hp in seen_patch:
            dropped.append((d.instance_id, seen_patch[hp], "patch"))
            continue
        if hi and hi in seen_issue:
            dropped.append((d.instance_id, seen_issue[hi], "problem_statement"))
            continue
        if hp:
            seen_patch[hp] = d.instance_id
        if hi:
            seen_issue[hi] = d.instance_id
        kept.append(d)
    return kept, dropped


_MERSENNE = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash_signature(sh: set[str], num_perm: int = 128, seed: int = 1) -> list[int]:
    rng = random.Random(seed)
    params = [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE)) for _ in range(num_perm)]
    if not sh:
        return [_MERSENNE] * num_perm
    base = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
    return [min((a * x + b) % _MERSENNE for x in base) for a, b in params]


def estimate_jaccard(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(1, len(a))


def _near_dup_text(d: Demo) -> str:
    return f"{d.problem_statement}\n{d.patch}"


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def semantic_dup_pairs(demos: list[Demo], embedder, threshold: float = 0.92,
                       patch_threshold: float = 0.97) -> list[tuple[str, str, float]]:
    """(id_a, id_b, cosine) for instances that are the same bug in different words.

    Issue and fix are embedded separately, because a terse re-report of a long issue scores
    lower against it than unrelated issues do against each other. A pair is flagged when the
    issues match (>= threshold) or the fixes match (>= patch_threshold: same change with renamed
    variables / reformatting, which exact patch dedup misses). `embedder`: spec or Embedder."""
    from .embed import make_embedder, similar_pairs

    emb = make_embedder(embedder) if isinstance(embedder, str) else embedder
    ids = [d.instance_id for d in demos]
    best: dict[tuple[int, int], float] = {}
    for texts, t in (([d.problem_statement for d in demos], threshold),
                     ([_hunk_text(d.patch) or d.problem_statement for d in demos], patch_threshold)):
        for i, j, c in similar_pairs(emb.embed(texts), t):
            best[(i, j)] = max(best.get((i, j), 0.0), c)
    return [(ids[i], ids[j], round(c, 4)) for (i, j), c in sorted(best.items())]


def _hunk_text(patch: str) -> str:
    """Patch hunks with context lines, minus headers/hashes. Context matters: on the seed set the
    changed lines alone scored a renamed copy of a fix at 0.90, below unrelated pairs (0.94)."""
    return "\n".join(ln for ln in (patch or "").splitlines()
                     if not ln.startswith(("diff ", "index ", "--- ", "+++ ")))


def near_dup_clusters(demos: list[Demo], threshold: float = 0.8, num_perm: int = 128,
                      k: int = 5, pairwise_max: int = 1500,
                      semantic_pairs: Iterable[tuple[str, str, float]] = ()) -> dict[str, str]:
    """MinHash over word k-shingles of problem_statement + patch, union-find into clusters.

    Pairwise estimate for small N, LSH banding (candidate generation, then verification) for
    large N. `semantic_pairs` (from semantic_dup_pairs) are unioned in too, so paraphrased
    duplicates share a cluster and the repo/cluster-grouped split keeps them on one side.
    Returns instance_id -> cluster_id ("cluster:<smallest member id>")."""
    sigs = [minhash_signature(shingles(_near_dup_text(d), k), num_perm) for d in demos]
    n = len(demos)
    uf = _UF(n)
    if n <= pairwise_max:
        pairs: Iterable[tuple[int, int]] = ((i, j) for i in range(n) for j in range(i + 1, n))
    else:
        # choose rows r so the LSH S-curve midpoint (1/b)^(1/r) sits a bit below the threshold
        best = min(((b, num_perm // b) for b in range(1, num_perm + 1) if num_perm % b == 0),
                   key=lambda br: abs((1 / br[0]) ** (1 / br[1]) - 0.85 * threshold))
        b, r = best
        cand: set[tuple[int, int]] = set()
        for band in range(b):
            buckets: dict[tuple, list[int]] = defaultdict(list)
            for i, s in enumerate(sigs):
                buckets[tuple(s[band * r:(band + 1) * r])].append(i)
            for members in buckets.values():
                cand.update((members[x], members[y]) for x in range(len(members))
                            for y in range(x + 1, len(members)))
        pairs = sorted(cand)
    for i, j in pairs:
        if estimate_jaccard(sigs[i], sigs[j]) >= threshold:
            uf.union(i, j)
    pos = {d.instance_id: i for i, d in enumerate(demos)}
    for a, b, _ in semantic_pairs:
        if a in pos and b in pos:
            uf.union(pos[a], pos[b])
    groups: dict[int, list[str]] = defaultdict(list)
    for i, d in enumerate(demos):
        groups[uf.find(i)].append(d.instance_id)
    out = {}
    for members in groups.values():
        cid = f"cluster:{min(members)}"
        for m in members:
            out[m] = cid
    return out


# ============================================================================================
# sizes
# ============================================================================================
def patch_size(patch: str) -> dict[str, Any]:
    files, hunks, lines = set(), 0, 0
    for line in (patch or "").splitlines():
        m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
        if m:
            files.add(m.group(2))
        elif line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if line.startswith("+++ ") and p != "/dev/null":
                files.add(p[2:] if p.startswith("b/") else p)
        elif line.startswith("@@"):
            hunks += 1
        elif line[:1] in "+-":
            lines += 1
    bucket = "multi-file" if len(files) > 1 else "multi-hunk" if hunks > 1 else "single-hunk"
    return {"lines": lines, "files": len(files), "hunks": hunks, "bucket": bucket}


def approx_tokens(demo: Demo, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    chars = len(demo.problem_statement)
    for s in demo.trajectory:
        chars += len(s.thought) + len(s.observation) + len(json.dumps(s.action.args))
    return int(chars / chars_per_token)


def trajectory_stats(demo: Demo, max_steps: int = MAX_STEPS, max_tokens: int = MAX_TOKENS,
                     chars_per_token: float = CHARS_PER_TOKEN) -> dict[str, Any]:
    toks = approx_tokens(demo, chars_per_token)
    flags = []
    if len(demo.trajectory) > max_steps:
        flags.append(f"steps>{max_steps}")
    if toks > max_tokens:
        flags.append(f"tokens>{max_tokens}")
    return {"steps": len(demo.trajectory), "approx_tokens": toks, "flags": flags}


# ============================================================================================
# test leakage
# ============================================================================================
MIN_LEAK_BLOCK = 2
_COMPARISON = re.compile(r"==|!=|<=|>=|(?<![-=])[<>](?!=)|\b(?:not in|in|is not|is)\b|assert(?:Equal|In|Is|True|False|Raises)")
_TRIVIAL_TEST_LINE = re.compile(r"^(import |from \S+ import |@|def |class |return$|pass$|#)")


def _strip_assert_msg(expr: str) -> str:
    """`a == b, "msg"` -> `a == b` (first top-level comma)."""
    depth, quote = 0, None
    for i, ch in enumerate(expr):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return expr[:i].strip()
    return expr.strip()


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _test_func_names(test_ids: list[str]) -> list[str]:
    names = []
    for t in test_ids:
        name = t.split("::")[-1].split("[")[0].strip()
        name = name.split()[0] if name else name  # django-style "test_x (pkg.Class)"
        if name:
            names.append(name)
    return names


def hidden_test_files(demo: Demo) -> list[str]:
    """Files created by test_patch: they do not exist at inference time."""
    return patch_files(demo.test_patch, created_only=True)


def _mentions(text: str, path: str) -> bool:
    base = posixpath.basename(path)
    return path in text or bool(re.search(rf"(?<![\w./-]){re.escape(base)}(?![\w-])", text))


def hidden_test_read_steps(demo: Demo) -> list[int]:
    """Indices of steps whose *action* reads/runs a file created by test_patch (and not created
    earlier by the trajectory itself)."""
    hidden = hidden_test_files(demo)
    if not hidden:
        return []
    own: set[str] = set()
    idx = []
    for i, s in enumerate(demo.trajectory):
        a = s.action.args or {}
        path = posixpath.normpath(str(a.get("path", ""))) if a.get("path") else ""
        if s.action.tool == "create_file" and path:
            own.add(path)
            continue
        live = [h for h in hidden if h not in own]
        hit = False
        if s.action.tool in ("view_file", "search") and path:
            hit = any(path == h or (s.action.tool == "search" and path == h) for h in live)
        elif s.action.tool == "bash":
            cmd = str(a.get("command", ""))
            hit = any(_mentions(cmd, h) for h in live)
        elif s.action.tool == "run_tests":
            hit = any(_mentions(str(a.get("target") or ""), h) for h in live)
        if hit:
            idx.append(i)
    return idx


def detect_leakage(demo: Demo) -> list[str]:
    findings: list[str] = []
    issue = demo.problem_statement or ""
    issue_sq = _squash(issue)
    # (a) assertions with an expected value (`x == y`, `x in y`, ...) quoted in full; a bare call
    # like `assert f(2000)` or a repro line shared with the issue is normal issue writing.
    # (b) a block of >= MIN_LEAK_BLOCK consecutive non-trivial test lines copied into the issue.
    run: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if len(run) >= MIN_LEAK_BLOCK:
            findings.append(f"issue contains {len(run)} consecutive hidden test lines, e.g. {run[0][:100]!r}")
        run.clear()

    for line in demo.test_patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            flush()
            continue
        code = line[1:].strip()
        if not code or _TRIVIAL_TEST_LINE.match(code):
            flush()
            continue
        if code.startswith("assert ") or code.startswith("self.assert"):
            expr = code[len("assert "):] if code.startswith("assert ") else code
            expr = _strip_assert_msg(expr) if code.startswith("assert ") else expr
            sq = _squash(expr)
            if _COMPARISON.search(expr) and len(sq) >= 8 and sq in issue_sq and sq not in seen:
                seen.add(sq)
                findings.append(f"issue contains hidden assertion: {expr[:100]!r}")
        sq = _squash(code)
        if len(sq) >= 8 and sq in issue_sq:
            run.append(code)
        else:
            flush()
    flush()
    for name in _test_func_names(demo.FAIL_TO_PASS):
        if re.search(rf"\b{re.escape(name)}\b", issue):
            findings.append(f"issue names FAIL_TO_PASS test: {name}")
    hidden = hidden_test_files(demo)
    for h in hidden:
        if _mentions(issue, h):
            findings.append(f"issue names hidden test file: {h}")
    for i in hidden_test_read_steps(demo):
        s = demo.trajectory[i]
        findings.append(f"trajectory[{i}] {s.action.tool} reads hidden test file")
    for i, s in enumerate(demo.trajectory):
        if s.action.tool == "create_file":
            p = posixpath.normpath(str(s.action.args.get("path", "")))
            if p in hidden:
                findings.append(f"trajectory[{i}] creates a file at a hidden test path: {p}")
    return findings


def remove_hidden_test_reads(demo: Demo, root: str | Path | None = None,
                             backend: str = "auto") -> Demo:
    """Copy of `demo` without steps that read hidden test files. The trajectory changed, so the
    copy's replay status is unknown: with `root` it is replayed (observations refreshed from the
    re-execution, metadata.replay_ok set); without it metadata.replay_ok is cleared."""
    out = copy.deepcopy(demo)
    drop = set(hidden_test_read_steps(demo))
    if not drop:
        return out
    out.trajectory = [s for i, s in enumerate(out.trajectory) if i not in drop]
    out.metadata["removed_hidden_test_steps"] = sorted(drop)
    out.metadata.pop("replay_ok", None)
    if root is not None:
        r = replay(out, root, backend)
        out.metadata["replay_ok"] = r.ok
        if len(r.observations) == len(out.trajectory):
            for s, o in zip(out.trajectory, r.observations):
                s.observation = o
    return out


# ============================================================================================
# secrets / PII
# ============================================================================================
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----)?")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9]+-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe_key", re.compile(r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://[^\s/:@'\"]{1,64}:[^\s/@'\"]{3,128}@[\w.-]+")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*")),
    ("api_key", re.compile(
        r"""(?ix)\b(?:aws_secret_access_key|api[_-]?key|apikey|secret(?:[_-]?key)?|access[_-]?token|
        auth[_-]?token|token|password|passwd|pwd)\b["']?\s*[:=]\s*["']?
        [A-Za-z0-9_\-./+]{8,}(?![A-Za-z0-9_\-./+(\[])""")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    ("ip_address", re.compile(rf"(?<![\d.]){_OCTET}\.{_OCTET}\.{_OCTET}\.{_OCTET}(?![\d.])")),
]
_IP_ALLOW = re.compile(r"^(127\.|0\.0\.0\.0$|255\.255\.255\.255$)")
_EMAIL_ALLOW = re.compile(r"@(example\.(com|org|net)|localhost)$", re.I)


_URL_CRED_ALLOW = re.compile(r"://(user(name)?|u|me|admin|x|foo):(pass(word)?|secret|p|x|bar|\*+|<[^>]*>|\$\{?\w+\}?)@", re.I)


_RESERVED_TLD = re.compile(r"\.(test|example|invalid|localhost|local)$", re.I)


def _placeholder_email(addr: str) -> bool:
    """Fixture addresses (ann@x.io, a@b.io, *@example.com, *.test) are not PII."""
    local, _, domain = addr.rpartition("@")
    labels = domain.split(".")
    return bool(_EMAIL_ALLOW.search(addr) or _RESERVED_TLD.search(domain)
                or (len(local) <= 3 and len(labels) == 2 and len(labels[0]) <= 2))


def _allowed(kind: str, match: str) -> bool:
    return ((kind == "ip_address" and (bool(_IP_ALLOW.match(match))
                                       or all(len(o) == 1 for o in match.split("."))))  # 1.4.2.1 = a version
            or (kind == "email" and _placeholder_email(match))
            or (kind == "url_credentials" and bool(_URL_CRED_ALLOW.search(match))))


# ---- entropy: catches credentials in formats no regex above knows ----------------------------
_ENTROPY_TOKEN = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_=-]{20,200}(?![A-Za-z0-9+/_=-])")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SECRET_CONTEXT = re.compile(r"(?i)(secret|token|passw|pwd|api[_-]?key|apikey|credential|private|auth|bearer|signing)")
B64_ENTROPY = 4.2     # bits/char; random base64 of length 20-40 sits around 4.3-5.2, English/code < 4
HEX_ENTROPY = 3.0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def entropy_hits(text: str) -> list[tuple[int, int]]:
    """Spans of high-entropy tokens that look like credentials.

    Base64-like tokens need mixed case + digits and entropy >= B64_ENTROPY. Pure hex (git SHAs,
    hashes, which trajectories are full of) only counts on a line that also names a secret
    (`token = "9f86d0..."`). UUIDs, paths and dotted identifiers are ignored."""
    out = []
    for m in _ENTROPY_TOKEN.finditer(text or ""):
        tok = m.group(0).strip("=")
        if len(tok) < 20 or _UUID.match(tok) or tok.count("/") >= 2 or tok.count("_") >= 3 or tok.count("-") >= 3:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end != -1 else len(text)]
        if _HEX.match(tok):
            if len(tok) >= 32 and shannon_entropy(tok) >= HEX_ENTROPY and _SECRET_CONTEXT.search(line):
                out.append(m.span())
            continue
        classes = sum(bool(re.search(rx, tok)) for rx in (r"[a-z]", r"[A-Z]", r"[0-9]"))
        if classes == 3 and shannon_entropy(tok) >= B64_ENTROPY and len(re.findall(r"[0-9]", tok)) >= 2:
            out.append(m.span())
    return out


# ---- optional external scanner (pip install detect-secrets) ----------------------------------
SECRET_SCANNER = "builtin"   # builtin | auto | detect-secrets (set by `audit --secret-scanner`)


def _detect_secrets_kinds(text: str) -> list[str]:
    """Run detect-secrets' regular file scan (with its false-positive filters) over `text`."""
    try:
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import default_settings
    except ImportError:
        if SECRET_SCANNER == "detect-secrets":
            raise RuntimeError("--secret-scanner detect-secrets needs `pip install detect-secrets`")
        return []
    import os
    import tempfile

    if not (text or "").strip():
        return []
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="audit-scan-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        coll = SecretsCollection()
        with default_settings():
            coll.scan_file(path)
        kinds = {"detect_secrets:" + re.sub(r"\W+", "_", sec.type.lower()).strip("_")
                 for _, sec in coll}
    finally:
        os.unlink(path)
    return sorted(kinds)


def scan_text(text: str) -> list[str]:
    kinds = []
    for kind, rx in SECRET_PATTERNS:
        if any(not _allowed(kind, m.group(0)) for m in rx.finditer(text or "")):
            kinds.append(kind)
    if entropy_hits(text or ""):
        kinds.append("high_entropy_string")
    if SECRET_SCANNER != "builtin":
        kinds += [k for k in _detect_secrets_kinds(text) if k not in kinds]
    return kinds


def _demo_text_fields(demo: Demo) -> Iterable[tuple[str, str]]:
    yield "problem_statement", demo.problem_statement
    yield "patch", demo.patch
    yield "test_patch", demo.test_patch
    for i, s in enumerate(demo.trajectory):
        yield f"trajectory[{i}].thought", s.thought
        yield f"trajectory[{i}].action", json.dumps(s.action.args, ensure_ascii=False)
        yield f"trajectory[{i}].observation", s.observation


def scan_secrets(demo: Demo) -> list[str]:
    """Findings like 'aws_access_key in patch' (never the secret value itself)."""
    return [f"{kind} in {name}" for name, text in _demo_text_fields(demo) for kind in scan_text(text)]


def redact_text(text: str) -> str:
    for kind, rx in SECRET_PATTERNS:
        text = rx.sub(lambda m, k=kind: m.group(0) if _allowed(k, m.group(0)) else f"[REDACTED:{k}]", text)
    for a, b in reversed(entropy_hits(text)):
        text = text[:a] + "[REDACTED:high_entropy_string]" + text[b:]
    return text


def _redact_obj(v: Any) -> Any:
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, list):
        return [_redact_obj(x) for x in v]
    if isinstance(v, dict):
        return {k: _redact_obj(x) for k, x in v.items()}
    return v


def redact(demo: Demo) -> Demo:
    """Redacted copy. Note: redacting inside `patch`/`test_patch` can stop them applying, so a
    redacted demo must be re-validated before it is used as an environment."""
    out = copy.deepcopy(demo)
    out.problem_statement = redact_text(out.problem_statement)
    out.patch = redact_text(out.patch)
    out.test_patch = redact_text(out.test_patch)
    for s in out.trajectory:
        s.thought = redact_text(s.thought)
        s.observation = redact_text(s.observation)
        s.action.args = _redact_obj(s.action.args)
    out.metadata["secrets"] = []
    out.metadata["redacted"] = True
    return out


# ============================================================================================
# optional LLM labels
# ============================================================================================
BUG_TYPES = ["off-by-one", "null-handling", "api-misuse", "concurrency", "config", "logic-error",
             "wrong-operator", "error-handling", "state-update", "type-error", "boundary-condition",
             "normalization", "performance", "resource-leak", "other"]

_LABEL_SYSTEM = ("You classify software bugs. Answer with exactly one label from the given list "
                 "on a line of the form 'LABEL: <label>'.")
_JUDGE_SYSTEM = ("You audit bug-fix benchmark instances (SWE-bench Verified style). Decide whether "
                 "the issue text alone is sufficient for a competent engineer to determine what a "
                 "correct fix must do, without seeing the tests. Answer on the last line with "
                 "'VERDICT: SUFFICIENT' or 'VERDICT: INSUFFICIENT'.")


def _clip(s: str, n: int = 8000) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def _norm_label(s: str) -> str:
    return re.sub(r"[\s_]+", "-", s.strip().strip("`'\".*").lower())


def parse_bug_type(answer: str, taxonomy: list[str] = BUG_TYPES) -> str:
    tax = {_norm_label(t): t for t in taxonomy}
    m = re.search(r'"bug_type"\s*:\s*"([^"]+)"', answer or "")
    if m and _norm_label(m.group(1)) in tax:
        return tax[_norm_label(m.group(1))]
    for m in re.finditer(r"(?im)^\W*(?:label|bug[_ -]?type)\s*[:=]\s*(.+)$", answer or ""):
        lab = _norm_label(m.group(1))
        if lab in tax:
            return tax[lab]
    text = _norm_label(answer or "")
    hits = [(text.find(k), -len(k), v) for k, v in tax.items() if k in text]
    return min(hits)[2] if hits else ("other" if "other" in tax else taxonomy[-1])


def label_bug_type(demo: Demo, llm, taxonomy: list[str] = BUG_TYPES) -> str:
    prompt = (f"Labels: {', '.join(taxonomy)}\n\n<issue>\n{_clip(demo.problem_statement, 4000)}\n"
              f"</issue>\n\n<fix>\n{_clip(demo.patch)}\n</fix>\n\nWhich label fits the bug best?")
    return parse_bug_type(llm(_LABEL_SYSTEM, prompt), taxonomy)


def parse_sufficiency(answer: str) -> bool | None:
    a = answer or ""
    m = re.search(r'"sufficient"\s*:\s*(true|false)', a, re.I)
    if m:
        return m.group(1).lower() == "true"
    verdicts = re.findall(r"(?i)verdict\s*[:=]\s*\**\s*(insufficient|underspecified|sufficient|yes|no)", a)
    if verdicts:
        return verdicts[-1].lower() in ("sufficient", "yes")
    low = a.lower()
    if re.search(r"\b(insufficient|underspecified|not sufficient|ambiguous)\b", low):
        return False
    if re.search(r"\bsufficient\b", low):
        return True
    first = low.strip().split()[:1]
    return {"yes": True, "no": False}.get(first[0].strip(".,!") if first else "", None)


def judge_issue_sufficiency(demo: Demo, llm) -> bool | None:
    prompt = (f"<issue>\n{_clip(demo.problem_statement, 6000)}\n</issue>\n\n"
              f"For reference only (the solver will NOT see these):\n<gold_patch>\n{_clip(demo.patch)}\n"
              f"</gold_patch>\n<hidden_tests>\n{_clip(demo.test_patch)}\n</hidden_tests>\n\n"
              "Is the issue alone sufficient to determine what a correct fix must do?")
    try:
        return parse_sufficiency(llm(_JUDGE_SYSTEM, prompt))
    except Exception as e:  # a refusal/outage should not kill the audit
        log.warning("sufficiency judge failed on %s: %s", demo.instance_id, e)
        return None


# ============================================================================================
# driver + report
# ============================================================================================
def _steps_bucket(n: int) -> str:
    for hi, lab in ((0, "0"), (5, "1-5"), (10, "6-10"), (20, "11-20"), (40, "21-40")):
        if n <= hi:
            return lab
    return ">40"


@dataclass
class AuditReport:
    n_input: int = 0
    stages: list[tuple[str, int]] = field(default_factory=list)       # (filter, count after)
    per_repo: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropped: list[dict[str, str]] = field(default_factory=list)      # {instance_id, reason, detail}
    flags: dict[str, list[str]] = field(default_factory=dict)         # flag -> instance ids
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)
    clusters: dict[str, list[str]] = field(default_factory=dict)      # multi-member clusters only
    warnings: list[str] = field(default_factory=list)
    runs: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_markdown(self) -> str:
        L = ["# Demo audit report", "",
             f"Input instances: **{self.n_input}**; validation runs per instance: {self.runs}.", ""]
        L += ["## Counts after each filter", "", "| Stage | Remaining |", "|---|---:|",
              f"| input | {self.n_input} |"]
        L += [f"| {name} | {n} |" for name, n in self.stages]
        L.append("")
        if self.warnings:
            L += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + [""]
        L += ["## Per-repo table", "",
              "| Repo group | Total | Invalid | Flaky | Dup | Kept | Validation drop rate | Share of kept |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for repo, r in sorted(self.per_repo.items()):
            L.append(f"| {repo} | {r['total']} | {r['invalid']} | {r['flaky']} | {r['dup']} | "
                     f"{r['kept']} | {r['drop_rate']:.0%} | {r['share']:.1%} |")
        L.append("")
        L += ["## Flags (kept, not dropped)", "", "| Flag | Count | Instances |", "|---|---:|---|"]
        for name, ids in self.flags.items():
            shown = ", ".join(ids[:10]) + (f", ... (+{len(ids) - 10})" if len(ids) > 10 else "")
            L.append(f"| {name} | {len(ids)} | {shown} |")
        L.append("")
        L += ["## Distributions (kept)", ""]
        for name, dist in self.distributions.items():
            L += [f"**{name}**", "", "| Value | Count |", "|---|---:|"]
            L += [f"| {k} | {v} |" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], str(kv[0])))]
            L.append("")
        L += ["## Near-duplicate clusters", ""]
        if self.clusters:
            L += [f"- `{cid}`: {', '.join(m)}" for cid, m in sorted(self.clusters.items())]
        else:
            L.append("None (every kept instance is its own cluster).")
        L.append("")
        if self.dropped:
            L += ["## Dropped instances", "", "| Instance | Reason | Detail |", "|---|---|---|"]
            for d in self.dropped:
                detail = d["detail"].replace("|", "\\|").replace("\n", " ")[:200]
                L.append(f"| {d['instance_id']} | {d['reason']} | {detail} |")
            L.append("")
        return "\n".join(L)


def audit_demos(demos: list[Demo], root: str | Path = ".", runs: int = 3, llm=None,
                near_dup_threshold: float = 0.8, *, replay_trajectories: bool = True,
                public_map: dict[str, bool] | None = None, relabel: bool = False,
                grade_fn: GradeFn | None = None,
                replay_fn: Callable[[Demo, str | Path], bool] | None = None,
                max_steps: int = MAX_STEPS, max_tokens: int = MAX_TOKENS,
                embedder=None, semantic_threshold: float = 0.92, patch_threshold: float = 0.97
                ) -> tuple[list[Demo], AuditReport]:
    """Run the §2.1 checklist. `runs=0` skips execution-based validation (flaky=None).
    Input demos are not mutated; kept demos are enriched copies."""
    demos = [copy.deepcopy(d) for d in demos]
    rep = AuditReport(n_input=len(demos), runs=runs)
    per_repo: dict[str, Counter] = defaultdict(Counter)
    for d in demos:
        d.metadata.setdefault("repo_group", d.repo)
        per_repo[repo_group(d)]["total"] += 1

    # 1-2. validate x runs: invalid (consistently failing) then flaky
    valid_stage, nonflaky = [], []
    flaky_dropped = []
    for d in demos:
        if runs <= 0:
            d.metadata["flaky"] = None
            valid_stage.append(d)
            continue
        ok, flaky, problems = validate_n(d, root, runs, grade_fn)
        d.metadata["flaky"] = flaky
        if flaky:
            flaky_dropped.append((d, problems))
            valid_stage.append(d)
        elif not ok:
            per_repo[repo_group(d)]["invalid"] += 1
            rep.dropped.append({"instance_id": d.instance_id, "reason": "invalid",
                                "detail": "; ".join(problems)})
        else:
            valid_stage.append(d)
    rep.stages.append(("valid (gold patch flips F2P, P2P green)", len(valid_stage)))
    flaky_ids = {d.instance_id for d, _ in flaky_dropped}
    for d, problems in flaky_dropped:
        per_repo[repo_group(d)]["flaky"] += 1
        rep.dropped.append({"instance_id": d.instance_id, "reason": "flaky",
                            "detail": f"results differ across {runs} runs; " + "; ".join(problems)})
    nonflaky = [d for d in valid_stage if d.instance_id not in flaky_ids]
    rep.stages.append((f"non-flaky ({runs}x)", len(nonflaky)))

    # 3. exact dedup
    kept, dups = exact_dedup(nonflaky)
    by_id = {d.instance_id: d for d in nonflaky}
    for iid, of, which in dups:
        per_repo[repo_group(by_id[iid])]["dup"] += 1
        rep.dropped.append({"instance_id": iid, "reason": "exact-duplicate",
                            "detail": f"same normalized {which} as {of}"})
    rep.stages.append(("exact dedup (patch, problem_statement)", len(kept)))

    # 4. annotate kept
    flags: dict[str, list[str]] = {k: [] for k in
                                   ("replay_failed", "leakage", "secrets", "too_many_steps", "too_many_tokens")}
    sem_pairs = semantic_dup_pairs(kept, embedder, semantic_threshold, patch_threshold) if embedder else []
    sem_by_id: dict[str, list] = defaultdict(list)
    for a, b, c in sem_pairs:
        sem_by_id[a].append([b, c])
        sem_by_id[b].append([a, c])
    if embedder:
        rep.stages.append((f"semantic near-dup pairs (issue cos >= {semantic_threshold} or "
                           f"fix cos >= {patch_threshold})", len(sem_pairs)))
    clusters = near_dup_clusters(kept, near_dup_threshold, semantic_pairs=sem_pairs)
    for d in kept:
        md = d.metadata
        md["cluster_id"] = clusters[d.instance_id]
        if embedder:
            md["semantic_dups"] = sem_by_id.get(d.instance_id, [])
        md["patch_size"] = patch_size(d.patch)
        ts = trajectory_stats(d, max_steps, max_tokens)
        md["length_flags"] = ts["flags"]
        if any(f.startswith("steps>") for f in ts["flags"]):
            flags["too_many_steps"].append(d.instance_id)
        if any(f.startswith("tokens>") for f in ts["flags"]):
            flags["too_many_tokens"].append(d.instance_id)
        md["leakage"] = detect_leakage(d)
        if md["leakage"]:
            flags["leakage"].append(d.instance_id)
        md["secrets"] = scan_secrets(d)
        if md["secrets"]:
            flags["secrets"].append(d.instance_id)
        if replay_trajectories:
            if not d.trajectory:
                md["replay_ok"] = False
            else:
                md["replay_ok"] = bool((replay_fn or replay_check)(d, root))
            if not md["replay_ok"]:
                flags["replay_failed"].append(d.instance_id)
        if public_map is not None:
            for key in (d.instance_id, repo_group(d), d.repo):
                if key in public_map:
                    md["public"] = bool(public_map[key])
                    break
        md.setdefault("public", None)
        if llm is not None:
            if relabel or not md.get("bug_type"):
                try:
                    md["bug_type"] = label_bug_type(d, llm)
                except Exception as e:
                    log.warning("bug-type labeling failed on %s: %s", d.instance_id, e)
            md["issue_sufficient"] = judge_issue_sufficiency(d, llm)
        md.setdefault("issue_sufficient", None)
    rep.flags = flags

    # report tables
    kept_by_repo = Counter(repo_group(d) for d in kept)
    for repo, c in per_repo.items():
        validated = c["total"]
        dropped_val = c["invalid"] + c["flaky"]
        rate = dropped_val / validated if validated else 0.0
        share = kept_by_repo[repo] / len(kept) if kept else 0.0
        rep.per_repo[repo] = {"total": c["total"], "invalid": c["invalid"], "flaky": c["flaky"],
                              "dup": c["dup"], "kept": kept_by_repo[repo], "drop_rate": rate, "share": share}
        if rate > REPO_DROP_WARN:
            rep.warnings.append(
                f"repo `{repo}`: validation drop rate {rate:.0%} (> {REPO_DROP_WARN:.0%}); the env build "
                "for this repo is likely broken: fix the env, don't drop the repo")
        if len(kept) >= 50 and share > REPO_SHARE_WARN:
            rep.warnings.append(f"repo `{repo}` is {share:.0%} of kept instances (> {REPO_SHARE_WARN:.0%}); "
                                "cap it when building the train mix")
    rep.distributions = {
        "patch_size": dict(Counter(d.metadata["patch_size"]["bucket"] for d in kept)),
        "bug_type": dict(Counter(d.metadata.get("bug_type") or "unlabeled" for d in kept)),
        "steps": dict(Counter(_steps_bucket(len(d.trajectory)) for d in kept)),
        "difficulty": dict(Counter(d.metadata.get("difficulty") or "unknown" for d in kept)),
        "public": dict(Counter(str(d.metadata.get("public")) for d in kept)),
    }
    if llm is not None:
        rep.distributions["issue_sufficient"] = dict(Counter(str(d.metadata.get("issue_sufficient"))
                                                             for d in kept))
    members: dict[str, list[str]] = defaultdict(list)
    for d in kept:
        members[d.metadata["cluster_id"]].append(d.instance_id)
    rep.clusters = {c: sorted(m) for c, m in members.items() if len(m) > 1}
    return kept, rep


# ============================================================================================
# CLI
# ============================================================================================
def cmd_audit(a) -> int:
    demos = load_demos(a.demos)
    llm = None
    if a.label_model:
        from ..llm import make_text_llm
        llm = make_text_llm(a.label_model)
    public_map = json.loads(Path(a.public_map).read_text()) if a.public_map else None
    global SECRET_SCANNER
    SECRET_SCANNER = a.secret_scanner
    kept, rep = audit_demos(demos, root=a.root, runs=a.runs, llm=llm,
                            near_dup_threshold=a.near_dup_threshold,
                            replay_trajectories=not a.no_replay, public_map=public_map,
                            embedder=a.embedder, semantic_threshold=a.semantic_threshold)
    save_demos(kept, a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(rep.to_markdown())
    print(f"kept {len(kept)}/{len(demos)} -> {a.out}; report -> {a.report}")
    for w in rep.warnings:
        print(f"WARNING: {w}")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("audit", help="§2.1 audit: validate xN, flakiness, replay, dedup, "
                                            "leakage, secrets, sizes")
    p.add_argument("demos", help="input demos JSONL")
    p.add_argument("--out", default="runs/audit/demos.validated.jsonl")
    p.add_argument("--report", default="runs/audit/audit.md")
    p.add_argument("--root", default=".", help="dir that demo.repo paths are relative to")
    p.add_argument("--runs", type=int, default=3, help="validation runs per instance (0 = skip)")
    p.add_argument("--label-model", default=None,
                   help="TextLLM for bug_type labels + issue-sufficiency judge (e.g. claude-opus-5-5)")
    p.add_argument("--near-dup-threshold", type=float, default=0.8)
    p.add_argument("--embedder", default=None,
                   help="also cluster semantic near-duplicates: hash | st:MODEL | voyage:MODEL | openai:MODEL")
    p.add_argument("--semantic-threshold", type=float, default=0.92,
                   help="embedding cosine at/above which two instances count as the same bug")
    p.add_argument("--secret-scanner", choices=["builtin", "auto", "detect-secrets"], default="builtin",
                   help="builtin = provider regexes + entropy; auto/detect-secrets also run Yelp's "
                        "detect-secrets plugins (auto: only if installed)")
    p.add_argument("--no-replay", action="store_true", help="skip trajectory replay")
    p.add_argument("--public-map", default=None,
                   help="JSON {repo|repo_group|instance_id: true/false} for metadata.public")
    p.set_defaults(fn=cmd_audit)
