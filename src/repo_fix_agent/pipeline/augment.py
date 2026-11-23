"""§3h data augmentation: bug injection (P1), issue rephrasing (P2), repo-context perturbation (P2).

Bug injection (SWE-smith style)
-------------------------------
For a seed demo we build its *known-good* state (seed repo + gold ``patch`` + ``test_patch``) and
check that the full test suite passes. We then enumerate single-line AST mutations over the
non-test ``.py`` files (located with ``ast``, applied as surgical text edits on the node's
``(lineno, col_offset, end_lineno, end_col_offset)`` span so the diff is one line) and keep a
mutant only if >=1 test fails while the suite still runs (no syntax/collection error, no
timeout). For each kept mutant:

  * every test file that contains a failing test is *moved out of the repo* into the new
    instance's ``test_patch`` (the agent never sees the tests it is graded on); the other test
    files stay in the repo as ordinary existing tests;
  * ``FAIL_TO_PASS`` = the failing tests, ``PASS_TO_PASS`` = every other test;
  * ``patch`` = inverse diff (mutated -> known-good, source files only);
  * ``problem_statement`` = :func:`write_issue` (LLM if given, else an offline template);
  * ``trajectory=[]`` - synthetic instances are environments for teacher rollouts / RL, not BC data;
  * the instance is checked with ``grader.validate_instance`` and dropped if it fails.

Mutants with the same failing-test set in the same file are deduplicated. Candidate sites are
enumerated deterministically and sampled with ``random.Random(f"{seed}:{instance_id}")``.

Repo paths (important)
----------------------
Input demos' ``repo`` fields are resolved against ``root`` (``--root``, default cwd), exactly
like ``repo-fix validate``. Synthetic repos are written to ``out_repos_dir`` (``--repos-dir``,
default ``runs/repos_synth/``) and their ``repo`` field is stored *relative to root* when the
repos dir is inside root, otherwise as an absolute path (``Path(root) / abs`` == ``abs``). Either
way ``repo-fix validate OUT.jsonl --root <same root>`` works for every output record; rephrased
instances reuse their parent's ``repo`` value unchanged.

CLI
---
  repo-fix augment IN.jsonl --out OUT.jsonl --repos-dir runs/repos_synth [--max-per-demo 3]
      [--kinds mutation,rephrase,perturb] [--issue-model MODEL] [--seed 0] [--root .]
      [--exclude-ids FILE]

``rephrase`` needs ``--issue-model`` (skipped otherwise); mutation issues fall back to the
offline template without a model. ``--exclude-ids`` (frozen test set: JSON manifest with an
``instances`` mapping/list, a JSON list, or newline-separated ids) makes the run refuse to use
any listed instance as a seed.
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from ..grader import validate_instance
from ..schema import Demo
from . import repro

GIT_ID = ["-c", "user.name=augment", "-c", "user.email=augment@local", "-c", "commit.gpgsign=false"]
IGNORE = (".git", "__pycache__", ".pytest_cache", ".venv", "node_modules")
KINDS = ("mutation", "rephrase", "perturb")
STYLES = ("terse", "user-perspective", "stack-trace-only")

BUG_TYPES = {
    "compare_flip": "wrong-comparison",
    "const_off_by_one": "off-by-one",
    "slice_bound": "off-by-one",
    "range_minus_one": "off-by-one",
    "bool_op_swap": "logic-error",
    "drop_not": "logic-error",
    "arith_swap": "wrong-operator",
    "return_none": "missing-return-value",
    "remove_call": "missing-normalization",
    "get_to_index": "missing-key-handling",
}
OPERATORS = tuple(BUG_TYPES)


# ---- small utilities ---------------------------------------------------------------------------
def is_test_file(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    name = rel.rsplit("/", 1)[-1]
    return (rel.startswith("tests/") or "/tests/" in rel or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def _git(cwd: str | Path, *args: str, input: str | None = None) -> str:
    r = subprocess.run(["git", *GIT_ID, *args], cwd=cwd, capture_output=True, text=True, input=input,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def _copy_tree(src: str | Path, dst: str | Path) -> None:
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(*IGNORE))


def _read_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and not any(part in IGNORE for part in p.relative_to(root).parts):
            try:
                out[rel] = p.read_text()
            except UnicodeDecodeError:
                pass
    return out


def _diff(before: dict[str, str], after: dict[str, str]) -> str:
    """git-format diff turning file map `before` into `after` (only the given paths)."""
    with tempfile.TemporaryDirectory(prefix="augment-diff-") as tmp:
        _git(tmp, "init", "-q")
        for rel, txt in before.items():
            p = Path(tmp, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
        _git(tmp, "add", "-A")
        _git(tmp, "commit", "-q", "--allow-empty", "--no-verify", "-m", "before")
        for rel in before:
            if rel not in after:
                Path(tmp, rel).unlink()
        for rel, txt in after.items():
            p = Path(tmp, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
        _git(tmp, "add", "-A")
        return _git(tmp, "diff", "--cached", "HEAD")


def _apply(repo: Path, patch: str) -> None:
    if patch.strip():
        _git(repo, "apply", "--whitespace=nowarn", "-", input=patch)


def _resolve_repo(demo: Demo, root: str | Path) -> Path:
    if "://" in demo.repo:
        raise ValueError(f"{demo.instance_id}: only local repos can be augmented (got {demo.repo})")
    return Path(root) / demo.repo


def _repo_field(repo_dir: Path, root: str | Path) -> str:
    """Relative to root when inside it, else absolute (see module docstring)."""
    repo_dir, root_abs = repo_dir.resolve(), Path(root).resolve()
    try:
        return repo_dir.relative_to(root_abs).as_posix()
    except ValueError:
        return str(repo_dir)


def _check_excluded(demo: Demo, exclude_ids: set[str]) -> None:
    ids = {demo.instance_id, demo.metadata.get("parent_id")}
    hit = ids & set(exclude_ids)
    if hit:
        raise ValueError(f"refusing to use frozen/excluded instance as augmentation seed: {sorted(hit)}")


def load_exclude_ids(path: str | Path) -> set[str]:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}
    if isinstance(data, dict):
        data = data.get("instances", data.get("instance_ids", []))
    if isinstance(data, dict):
        return set(data)
    return {x["instance_id"] if isinstance(x, dict) else str(x) for x in data}


# ---- test running ----------------------------------------------------------------------------
_SUMMARY = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL) (\S+)")


@dataclass
class SuiteResult:
    returncode: int
    passed: set[str]
    failed: set[str]
    output: str
    timed_out: bool = False
    seconds: float = 0.0


def run_suite(repo: Path, timeout: int = 120) -> SuiteResult:
    """Run the whole pytest suite once in `repo` and parse per-test outcomes from `-rA`."""
    repo = Path(repo).resolve()  # rootdir must match cwd's realpath or node ids get mangled
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", "--tb=short",
            "--color=no", "--rootdir", str(repo)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    t0 = time.monotonic()
    try:
        r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return SuiteResult(124, set(), set(), out, True, time.monotonic() - t0)
    passed, failed = set(), set()
    for line in r.stdout.splitlines():
        m = _SUMMARY.match(line)
        if m:
            (passed if m.group(1) in ("PASSED", "XFAIL") else failed).add(m.group(2))
    return SuiteResult(r.returncode, passed, failed, r.stdout + r.stderr, False, time.monotonic() - t0)


# ---- mutation operators ------------------------------------------------------------------------
@dataclass(frozen=True)
class Mutation:
    operator: str
    file: str
    line: int           # 1-based
    col: int            # byte offsets within the line (as in ast)
    end_col: int
    replacement: str    # new text for line[col:end_col]

    def sort_key(self):
        return (self.file, self.line, self.col, self.end_col, self.operator, self.replacement)


def _seg(lines_b: list[bytes], line: int, col: int, end_col: int) -> str:
    return lines_b[line - 1][col:end_col].decode()


def apply_mutation(source: str, m: Mutation) -> str:
    lines = source.splitlines(keepends=True)
    b = lines[m.line - 1].encode()
    lines[m.line - 1] = (b[:m.col] + m.replacement.encode() + b[m.end_col:]).decode()
    return "".join(lines)


def mutated_lines(source: str, m: Mutation) -> tuple[str, str]:
    before = source.splitlines()[m.line - 1]
    after = apply_mutation(source, m).splitlines()[m.line - 1]
    return before, after


_ATOMIC = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)
_CMP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
_SEQ_LITERALS = (ast.List, ast.Tuple, ast.JoinedStr, ast.ListComp, ast.Dict, ast.Set)
_NORMALIZERS = {"lower", "upper", "strip", "lstrip", "rstrip", "casefold", "title"}


def _single(n: ast.AST) -> bool:
    return getattr(n, "lineno", None) is not None and n.lineno == n.end_lineno


def _is_str_or_seq(n: ast.AST) -> bool:
    return isinstance(n, _SEQ_LITERALS) or (isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)))


def enumerate_mutations(source: str, file: str = "<src>", operators: Iterable[str] = OPERATORS) -> list[Mutation]:
    """All single-line mutation candidates in `source`, deterministically ordered."""
    ops = set(operators)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines_b = [ln.encode() for ln in source.splitlines()]
    parents: dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out: list[Mutation] = []

    def text(n: ast.AST) -> str:
        return _seg(lines_b, n.lineno, n.col_offset, n.end_col_offset)

    def wrap(n: ast.AST) -> str:
        return text(n) if isinstance(n, _ATOMIC) else f"({text(n)})"

    def between(a: ast.AST, b: ast.AST, pattern: str, repl: str, op: str) -> None:
        """Replace the operator token found between the end of `a` and the start of `b`."""
        if a.end_lineno != b.lineno:
            return
        seg = _seg(lines_b, a.end_lineno, a.end_col_offset, b.col_offset)
        found = list(re.finditer(pattern, seg))
        if len(found) != 1:
            return
        f = found[0]
        new = seg[:f.start()] + repl + seg[f.end():]
        out.append(Mutation(op, file, a.end_lineno, a.end_col_offset, b.col_offset, new))

    in_branch = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            for c in ast.walk(n):
                if c is not n:
                    in_branch.add(id(c))
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and "compare_flip" in ops:
            left = n.left
            for op_node, right in zip(n.ops, n.comparators):
                sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
                       ast.NotEq: "!="}.get(type(op_node))
                if sym:
                    between(left, right, r"(?<![<>=!])" + re.escape(sym) + r"(?![<>=])", _CMP[sym], "compare_flip")
                left = right
        elif isinstance(n, ast.Constant) and "const_off_by_one" in ops:
            if type(n.value) is int and _single(n) and id(n) not in docstrings:
                for d in (1, -1):
                    v = n.value + d
                    if v >= 0:
                        out.append(Mutation("const_off_by_one", file, n.lineno, n.col_offset,
                                            n.end_col_offset, str(v)))
        elif isinstance(n, ast.Slice) and "slice_bound" in ops:
            for b in (n.lower, n.upper):
                if b is not None and _single(b) and not isinstance(b, ast.Constant):
                    for sign in ("+", "-"):
                        out.append(Mutation("slice_bound", file, b.lineno, b.col_offset, b.end_col_offset,
                                            f"{wrap(b)} {sign} 1"))
        elif isinstance(n, ast.BoolOp) and "bool_op_swap" in ops:
            src, dst = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
            for a, b in zip(n.values, n.values[1:]):
                between(a, b, rf"\b{src}\b", dst, "bool_op_swap")
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and "drop_not" in ops and _single(n):
            full = text(n)
            if full.startswith("not"):
                inner = full[3:].strip()
                if isinstance(n.operand, (ast.BoolOp, ast.Compare)) and not inner.startswith("("):
                    inner = f"({inner})"
                out.append(Mutation("drop_not", file, n.lineno, n.col_offset, n.end_col_offset, inner))
        elif isinstance(n, ast.BinOp) and "arith_swap" in ops and _single(n):
            seqs = _is_str_or_seq(n.left) or _is_str_or_seq(n.right)
            if isinstance(n.op, ast.Add) and not seqs:
                between(n.left, n.right, r"\+", "-", "arith_swap")
            elif isinstance(n.op, ast.Sub):
                between(n.left, n.right, r"-", "+", "arith_swap")
            elif isinstance(n.op, ast.Mult) and not seqs:
                between(n.left, n.right, r"(?<!\*)\*(?!\*)", "/", "arith_swap")
            elif isinstance(n.op, ast.Div):
                between(n.left, n.right, r"(?<!/)/(?!/)", "//", "arith_swap")
                between(n.left, n.right, r"(?<!/)/(?!/)", "*", "arith_swap")
            elif isinstance(n.op, ast.FloorDiv):
                between(n.left, n.right, r"//", "/", "arith_swap")
        elif isinstance(n, ast.Return) and "return_none" in ops:
            v = n.value
            if (v is not None and _single(v) and id(n) in in_branch
                    and not (isinstance(v, ast.Constant) and v.value is None)):
                out.append(Mutation("return_none", file, v.lineno, v.col_offset, v.end_col_offset, "None"))
        elif isinstance(n, ast.Call) and _single(n):
            f = n.func
            if ("remove_call" in ops and isinstance(f, ast.Attribute) and f.attr in _NORMALIZERS
                    and not n.args and not n.keywords and _single(f.value)):
                out.append(Mutation("remove_call", file, n.lineno, n.col_offset, n.end_col_offset, text(f.value)))
            if ("range_minus_one" in ops and isinstance(f, ast.Name) and f.id == "range"
                    and 1 <= len(n.args) <= 3 and not n.keywords):
                stop = n.args[0] if len(n.args) == 1 else n.args[1]
                if _single(stop):
                    out.append(Mutation("range_minus_one", file, stop.lineno, stop.col_offset,
                                        stop.end_col_offset, f"{wrap(stop)} - 1"))
            if ("get_to_index" in ops and isinstance(f, ast.Attribute) and f.attr == "get"
                    and 1 <= len(n.args) <= 2 and not n.keywords and _single(f.value)
                    and not any(isinstance(a, ast.Starred) for a in n.args)):
                out.append(Mutation("get_to_index", file, n.lineno, n.col_offset, n.end_col_offset,
                                    f"{wrap(f.value)}[{text(n.args[0])}]"))
    # keep only edits that still parse and actually change something
    good, seen = [], set()
    for m in sorted(set(out), key=Mutation.sort_key):
        new = apply_mutation(source, m)
        if new == source or (m.line, m.col, m.end_col, m.replacement) in seen:
            continue
        try:
            ast.parse(new)
        except SyntaxError:
            continue
        seen.add((m.line, m.col, m.end_col, m.replacement))
        good.append(m)
    return good


# ---- issue writing -----------------------------------------------------------------------------
ISSUE_SYSTEM = (
    "You write realistic GitHub bug reports as a user of a Python library would. Describe the "
    "observed vs expected behaviour of the public function(s), optionally with a tiny usage "
    "snippet. Never mention tests, test functions, test files, file paths, line numbers, or the "
    "exact line of code that is wrong, and never suggest the fix. Output only the issue: a "
    "one-line title, a blank line, then the body."
)

_TEST_NAME = re.compile(r"\btest_\w+")
_PATHISH = re.compile(r"[\w./-]*tests?/[\w./-]+|[\w./-]+\.py(:\d+)?(::\w+)?")


def _leaks(text: str, forbidden: Iterable[str]) -> str | None:
    if _TEST_NAME.search(text):
        return f"mentions test identifier {_TEST_NAME.search(text).group(0)!r}"
    low = text.lower()
    for f in forbidden:
        f = f.strip()
        if len(f) >= 4 and f.lower() in low:
            return f"mentions forbidden text {f!r}"
    return None


def _scrub(s: str) -> str:
    s = _TEST_NAME.sub("", s)
    s = _PATHISH.sub("", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def _first_failure(output: str) -> tuple[str | None, list[str]]:
    """(statement in the *test* frame, E-lines) of the first failure block in pytest --tb output.

    Only the first (outermost = test) frame's code line is used, so source lines - including the
    mutated one - shown in deeper frames are never quoted.
    """
    stmt, elines, started, frames = None, [], False, 0
    for ln in output.splitlines():
        if re.match(r"^_{3,} .* _{3,}$", ln):
            if started and (elines or stmt):
                break
            started, stmt, elines, frames = True, None, [], 0
            continue
        if not started or ln.startswith("=" * 5):
            if started and ln.startswith("=" * 5):
                break
            continue
        if ln.startswith("E "):
            elines.append(ln[1:].strip())
        elif re.match(r"^\S+:\d+: in \S+", ln):
            frames += 1
        elif ln.startswith(">") and frames <= 1 and stmt is None:   # --tb=long/auto
            stmt = ln[1:].strip()
        elif ln.startswith("    ") and ln.strip() and frames == 1 and stmt is None:
            stmt = ln.strip()
    return stmt, [e for e in elines if e]


_NOT_FUNCS = {"assert", "list", "dict", "set", "tuple", "len", "range", "sorted", "str", "int",
              "float", "approx", "raises", "pytest", "isinstance", "repr", "print", "sum", "any", "all",
              "bool", "type", "min", "max", "frozenset", "next", "iter", "map", "filter", "zip"}


def _calls_in(expr: str) -> list[str]:
    out = []
    for m in re.finditer(r"([A-Za-z_][\w.]*)\(", expr):
        name = m.group(1).split(".")[-1]
        if (name in _NOT_FUNCS or name.startswith("test_") or name.endswith(("Error", "Exception"))
                or (m.start() > 0 and expr[m.start() - 1] in ".")):
            continue
        depth = 0
        for j in range(m.end() - 1, len(expr)):
            depth += {"(": 1, ")": -1}.get(expr[j], 0)
            if depth == 0:
                out.append(expr[m.start():j + 1])
                break
    return out


def _pick_call(expr: str, known_funcs: Iterable[str]) -> str | None:
    calls = _calls_in(expr)
    known = set(known_funcs)
    for c in calls:
        if c.split("(")[0].split(".")[-1] in known:
            return c
    return calls[0] if calls and not known else None


def _top_level_name(source: str) -> str | None:
    m = re.search(r"^(?:async )?def (\w+)\(|^class (\w+)", source, re.M)
    return (m.group(1) or m.group(2)) if m else None


def fallback_issue(failing_output: str, source_context: str = "", *, known_funcs: Iterable[str] = (),
                   func_hint: str | None = None) -> str:
    """Offline template issue built from the first failure (test frame + assertion/exception)."""
    stmt, elines = _first_failure(failing_output)
    call = _pick_call(stmt or "", known_funcs) if stmt else None
    func = call.split("(")[0].split(".")[-1] if call else (func_hint or _top_level_name(source_context))
    target = f"`{call}`" if call else (f"`{func}`" if func else "The function")
    title = f"`{func}` returns wrong result" if func else "Unexpected result"
    no_raise = next((e for e in elines if "DID NOT RAISE" in e), None)
    exc = next((e for e in elines if re.match(r"^[\w.]*(Error|Exception)\b", e)
                and not e.startswith(("AssertionError", "Failed"))), None)
    cmp_line = next((e for e in elines if re.match(r"^(?:AssertionError: )?assert ", e)), None)
    where = next((e for e in elines if e.lstrip("+ ").startswith("where ")), None)
    m_eq = re.match(r"^assert (.+?) == (.+)$", stmt or "")
    m_cmp = re.match(r"^(?:AssertionError: )?assert (.+?) == (.+)$", cmp_line or "")
    if no_raise:
        m = re.search(r"DID NOT RAISE\s*(?:<class '([\w.]+)'>|(\w+))", no_raise)
        exc_name = (m.group(1) or m.group(2)).split(".")[-1] if m else "an exception"
        title = f"{target} does not raise {exc_name} for invalid input"
        body = f"{target} is expected to reject invalid arguments with `{exc_name}`, but it silently returns instead."
    elif exc:
        exc_name = exc.split(":")[0].split(".")[-1]
        title = f"`{func}` raises {exc_name}" if func else f"Unexpected {exc_name}"
        body = f"Calling {target} raises `{_scrub(exc)}`; it should not."
    elif m_eq and call and m_eq.group(1).strip() == call and m_cmp:
        body = f"Calling `{call}` returns `{m_cmp.group(1)}` but `{m_eq.group(2).strip()}` was expected."
    else:
        if cmp_line and re.match(r"^(?:AssertionError: )?assert (False|True|None)$", cmp_line) and where:
            m = re.match(r"^\+?\s*where (.+?) = (.+)$", where)
            observed = f"`{m.group(2)}` returned `{m.group(1)}`" if m else _scrub(where)
        else:
            observed = f"`{_scrub(cmp_line or (elines[0] if elines else ''))}`"
        body = f"{target} does not behave as expected. Observed: {observed}."
        # (no "expected the following to hold" + assertion: quoting the hidden test is leakage)
    return f"{_scrub(title)}\n\n{body}\n"


def write_issue(failing_output: str, source_context: str, llm: Callable[[str, str], str] | None = None,
                *, forbidden: Iterable[str] = (), retries: int = 1, known_funcs: Iterable[str] = (),
                func_hint: str | None = None, symptom: dict | None = None, seed: str = "") -> str:
    """Write a user-facing bug report; rejects LLM outputs that leak tests or the mutated line.

    With `llm`: up to `retries + 1` attempts, each checked by `_leaks` (test_* identifiers or any
    `forbidden` string, e.g. test names, hidden test paths, the mutated line); then fall back to
    the offline writer, which is itself re-checked and reduced to a bare title+summary if needed.

    `symptom` (from repro.find_symptom) is an executed reproduction with buggy vs fixed outcomes;
    when present it grounds the LLM prompt and drives the offline writer (repro.compose_issue)
    instead of the assertion-quoting template.
    """
    forbidden = list(forbidden)
    if llm is not None:
        grounding = (f"{repro.symptom_summary(symptom)}\n\n" if symptom else "")
        prompt = (f"{grounding}Failing behaviour (pytest output, for your eyes only):\n```\n{failing_output[-4000:]}\n```\n\n"
                  f"Relevant code:\n```python\n{source_context[:4000]}\n```\n\nWrite the bug report.")
        for _ in range(retries + 1):
            try:
                text = llm(ISSUE_SYSTEM, prompt).strip()
            except Exception:  # noqa: BLE001 - fall back on any model error
                break
            why = _leaks(text, forbidden) if text else "empty output"
            if why is None:
                return text + "\n"
            prompt += f"\n\nYour previous draft was rejected because it {why}. Rewrite it without that."
    text = None
    if symptom:
        text = repro.compose_issue(symptom, source_context=source_context, seed=seed)
        if _leaks(text, forbidden):
            text = None
    if text is None:
        text = fallback_issue(failing_output, source_context, known_funcs=known_funcs, func_hint=func_hint)
    if _leaks(text, forbidden):
        name = func_hint or _top_level_name(source_context)
        text = (f"`{name}` returns wrong result\n\n`{name}` produces incorrect results for normal inputs.\n"
                if name else "Incorrect result\n\nA library function produces incorrect results.\n")
    return text


# ---- bug injection -----------------------------------------------------------------------------
@dataclass
class AugmentStats:
    kept: Counter = field(default_factory=Counter)      # by operator / kind
    dropped: Counter = field(default_factory=Counter)   # by reason
    per_demo: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kept": dict(self.kept), "dropped": dict(self.dropped), "per_demo": self.per_demo}


def known_good(demo: Demo, root: str | Path, dest: Path) -> Path:
    """Materialise seed repo + gold patch + test_patch at `dest`."""
    _copy_tree(_resolve_repo(demo, root), dest)
    _git(dest, "init", "-q")
    _apply(dest, demo.patch)
    _apply(dest, demo.test_patch)
    shutil.rmtree(dest / ".git")
    return dest


def _enclosing_source(source: str, line: int) -> str:
    tree = ast.parse(source)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno:
            if best is None or n.lineno > best.lineno:
                best = n
    lines = source.splitlines()
    if best is None:
        return "\n".join(lines[max(0, line - 6):line + 5])
    start = min([d.lineno for d in best.decorator_list] + [best.lineno])
    return "\n".join(lines[start - 1:best.end_lineno])


def _defined_names(sources: dict[str, str]) -> set[str]:
    names = set()
    for txt in sources.values():
        try:
            tree = ast.parse(txt)
        except SyntaxError:
            continue
        names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and not n.name.startswith("_")}
    return names


def _outer_name(source: str, line: int) -> str | None:
    for n in ast.parse(source).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.lineno <= line <= n.end_lineno:
            return n.name
    return None


def _failing_block(output: str) -> str:
    """Keep the FAILURES section of pytest output (drop PASSED summary noise)."""
    keep = []
    for ln in output.splitlines():
        if ln.startswith(("PASSED ", "XFAIL ")):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _difficulty(op: str, n_f2p: int, n_src_files: int, test_imports_file: bool) -> str:
    score = (n_f2p <= 1) + (not test_imports_file) + (n_src_files > 3) + \
        (op in ("const_off_by_one", "compare_flip", "slice_bound", "bool_op_swap"))
    return "easy" if score <= 1 else "medium" if score == 2 else "hard"


def mutate_demo(demo: Demo, root: str | Path, out_repos_dir: str | Path, *, max_mutants: int = 3,
                operators: Iterable[str] = OPERATORS, llm: Callable[[str, str], str] | None = None,
                seed: int = 0, exclude_ids: set[str] = frozenset(), max_tries: int | None = None,
                validate: bool = True, stats: AugmentStats | None = None) -> list[Demo]:
    _check_excluded(demo, exclude_ids)
    stats = stats if stats is not None else AugmentStats()
    out_repos_dir = Path(out_repos_dir)
    max_tries = max_tries if max_tries is not None else max(10, 8 * max_mutants)
    new_demos: list[Demo] = []
    with tempfile.TemporaryDirectory(prefix="augment-") as tmp:
        try:
            good = known_good(demo, root, Path(tmp) / "good")
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            stats.dropped["known_good_build_failed"] += 1
            stats.per_demo[demo.instance_id] = 0
            print(f"[augment] {demo.instance_id}: cannot build known-good state: {e}", file=sys.stderr)
            return []
        base = run_suite(good)
        if base.returncode != 0 or base.failed or not base.passed:
            stats.dropped["known_good_fails"] += 1
            stats.per_demo[demo.instance_id] = 0
            return []
        all_tests = base.passed
        timeout = int(max(10, base.seconds * 10))
        files = _read_tree(good)
        buggy = Path(tmp) / "buggy"   # mutated twin of `good`, for executing reproductions
        _copy_tree(good, buggy)
        src_files = sorted(f for f in files if f.endswith(".py") and not is_test_file(f))
        cands = [m for f in src_files for m in enumerate_mutations(files[f], f, operators)]
        rng = random.Random(f"{seed}:{demo.instance_id}")
        rng.shuffle(cands)
        seen_sets: set[tuple[str, frozenset]] = set()
        tries = 0
        for m in cands:
            if len(new_demos) >= max_mutants or tries >= max_tries:
                break
            tries += 1
            orig = files[m.file]
            mutated = apply_mutation(orig, m)
            try:
                compile(mutated, m.file, "exec")
            except SyntaxError:
                stats.dropped["syntax_error"] += 1
                continue
            (good / m.file).write_text(mutated)
            try:
                res = run_suite(good, timeout=timeout)
            finally:
                (good / m.file).write_text(orig)
            failing = all_tests - res.passed
            if res.timed_out:
                stats.dropped["timeout"] += 1
                continue
            if res.returncode not in (0, 1):
                stats.dropped["suite_error"] += 1
                continue
            if not failing:
                stats.dropped["no_failing_tests"] += 1
                continue
            if failing == all_tests and re.search(r"\b(ImportError|ModuleNotFoundError|SyntaxError)\b", res.output):
                stats.dropped["import_error"] += 1
                continue
            key = (m.file, frozenset(failing))
            if key in seen_sets:
                stats.dropped["duplicate_failing_set"] += 1
                continue
            seen_sets.add(key)
            j = len(new_demos)
            (buggy / m.file).write_text(mutated)
            try:
                known = _defined_names({f: t for f, t in files.items() if f.endswith(".py") and not is_test_file(f)})
                tests_src = {f: files[f] for f in {t.split("::")[0] for t in failing} if f in files}
                symptom = repro.find_symptom(tests_src, failing, known, buggy, good)
            finally:
                (buggy / m.file).write_text(orig)
            d = _build_mutant(demo, m, files, mutated, failing, all_tests, res, j, root, out_repos_dir, llm,
                              symptom=symptom)
            problems = validate_instance(d, root) if validate else []
            if problems:
                stats.dropped["validate_failed"] += 1
                shutil.rmtree(_resolve_repo(d, root), ignore_errors=True)
                print(f"[augment] {d.instance_id}: dropped: {problems}", file=sys.stderr)
                seen_sets.discard(key)
                continue
            stats.kept[m.operator] += 1
            new_demos.append(d)
    stats.per_demo[demo.instance_id] = len(new_demos)
    return new_demos


def _source_defining(files: dict[str, str], name: str) -> str | None:
    rx = re.compile(rf"^\s*(?:async )?(?:def|class) {re.escape(name)}\b", re.M)
    return next((t for f, t in sorted(files.items())
                 if f.endswith(".py") and not is_test_file(f) and rx.search(t)), None)


def _unleak(issue: str, test_patch: str, f2p: list[str], name: str | None) -> str:
    """Final guard: run the audit's leakage detector on the issue; degrade until it is clean."""
    from .audit import detect_leakage  # local import: audit is heavier and imports grader

    def leaks(text: str) -> bool:
        probe = Demo(instance_id="probe", repo="", base_commit=None, problem_statement=text,
                     test_cmd="", test_patch=test_patch, FAIL_TO_PASS=f2p, PASS_TO_PASS=[], patch="")
        return bool(detect_leakage(probe))

    if not leaks(issue):
        return issue
    blocks = issue.split("\n\n")
    keep = []
    for i, blk in enumerate(blocks):   # drop code blocks and the "To reproduce:" line introducing them
        nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
        if blk.startswith("    ") or (blk.rstrip().endswith(":") and nxt.startswith("    ")):
            continue
        keep.append(blk)
    trimmed = "\n\n".join(keep)
    if trimmed.strip() and not leaks(trimmed):
        return trimmed.rstrip() + "\n"
    return (f"`{name}` returns wrong result\n\n`{name}` produces incorrect results for normal inputs.\n"
            if name else "Incorrect result\n\nA library function produces incorrect results.\n")


def _build_mutant(demo: Demo, m: Mutation, files: dict[str, str], mutated: str, failing: set[str],
                  all_tests: set[str], res: SuiteResult, j: int, root: str | Path, out_repos_dir: Path,
                  llm, symptom: dict | None = None) -> Demo:
    new_id = f"{demo.instance_id}-mut{j}"
    hidden_files = sorted({t.split("::")[0] for t in failing})
    repo_dir = out_repos_dir / new_id
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    for rel, txt in files.items():
        if rel in hidden_files:
            continue
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mutated if rel == m.file else txt)
    patch = _diff({m.file: mutated}, {m.file: files[m.file]})
    test_patch = _diff({}, {rel: files[rel] for rel in hidden_files})
    before, after = mutated_lines(files[m.file], m)
    test_funcs = sorted({t.split("::")[-1].split("[")[0] for t in all_tests})
    known = _defined_names({f: t for f, t in files.items() if f.endswith(".py") and not is_test_file(f)})
    ctx = _enclosing_source(mutated, m.line)
    if symptom and symptom["probe"].funcs:  # docstring of the function the reporter actually called
        ctx = _source_defining({**files, m.file: mutated}, symptom["probe"].funcs[0]) or ctx
    issue = write_issue(_failing_block(res.output), ctx, llm,
                        forbidden=[*test_funcs, *hidden_files, after.strip()], known_funcs=known,
                        func_hint=_outer_name(mutated, m.line), symptom=symptom, seed=new_id)
    f2p = sorted(failing)
    issue = _unleak(issue, test_patch, f2p, _outer_name(mutated, m.line))
    module = m.file[:-3].replace("/", ".")
    mod_tail = module.split(".")[-1]
    test_imports = any(mod_tail in files[h] for h in hidden_files)
    n_src = sum(1 for f in files if f.endswith(".py") and not is_test_file(f) and not f.endswith("__init__.py"))
    return Demo(
        instance_id=new_id, repo=_repo_field(repo_dir, root), base_commit=None, problem_statement=issue,
        test_cmd=demo.test_cmd, test_patch=test_patch, FAIL_TO_PASS=f2p,
        PASS_TO_PASS=sorted(all_tests - failing), patch=patch, trajectory=[],
        metadata={
            "source": "synthetic:mutation", "parent_id": demo.instance_id,
            "mutation": {"operator": m.operator, "file": m.file, "line": m.line, "before": before, "after": after},
            "bug_type": BUG_TYPES[m.operator],
            "difficulty": _difficulty(m.operator, len(f2p), n_src, test_imports),
            "files_changed": [m.file], "num_steps": 0, "language": demo.metadata.get("language", "python"),
            "issue_writer": "llm" if llm is not None else ("repro" if symptom else "template"),
            # hidden failing tests beyond the one symptom the issue shows: the "fix the example,
            # miss the rest of the contract" cases the baseline student fails on (docs/BASELINE.md)
            "hidden_beyond_issue": max(0, len(failing) - 1),
        },
    )


# ---- issue rephrasing --------------------------------------------------------------------------
STYLE_PROMPTS = {
    "terse": "Rewrite this bug report as a terse, one-to-three sentence report from a busy developer. "
             "Keep the key observed vs expected behaviour.",
    "user-perspective": "Rewrite this bug report from the perspective of an end user who noticed the symptom in "
                        "their application; describe what they did and saw, less about code internals.",
    "stack-trace-only": "Rewrite this bug report as little more than a minimal reproduction snippet and the "
                        "resulting error/traceback or wrong output, with at most one sentence of prose.",
}
REPHRASE_SYSTEM = ("You rewrite bug reports. Never mention test functions, test files or the fix. "
                   "Do not invent behaviour not implied by the original. Output only the rewritten issue.")


def _test_names(demo: Demo) -> list[str]:
    ids = demo.FAIL_TO_PASS + demo.PASS_TO_PASS
    return sorted({t.split("::")[-1].split("[")[0] for t in ids} | {t.split("::")[0] for t in ids})


def rephrase_issue(demo: Demo, llm: Callable[[str, str], str], style: str, *, retries: int = 1,
                   exclude_ids: set[str] = frozenset()) -> Demo:
    """Return a copy of `demo` with the issue rewritten in `style`; raises ValueError on leaks."""
    if style not in STYLE_PROMPTS:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    _check_excluded(demo, exclude_ids)
    prompt = f"{STYLE_PROMPTS[style]}\n\n<issue>\n{demo.problem_statement.strip()}\n</issue>"
    why = "no attempt"
    for _ in range(retries + 1):
        text = llm(REPHRASE_SYSTEM, prompt).strip()
        why = _leaks(text, _test_names(demo)) if text else "empty output"
        if why is None:
            return replace(demo, instance_id=f"{demo.instance_id}-reph-{style}", problem_statement=text + "\n",
                           trajectory=[], metadata={**demo.metadata, "source": "synthetic:rephrase",
                                                    "parent_id": demo.instance_id, "style": style,
                                                    "num_steps": 0})
        prompt += f"\n\nYour previous rewrite was rejected because it {why}. Try again."
    raise ValueError(f"{demo.instance_id}: rephrase ({style}) rejected: {why}")


def check_solvable(demo: Demo, policy_factory: Callable[[], Any], root: str | Path = ".",
                   max_steps: int = 40) -> bool:
    """Optional QA hook: run the agent once on `demo` and grade it. Not called by default."""
    from ..agent import run_agent
    from ..env import RepoEnv
    from ..grader import grade

    with RepoEnv(str(_resolve_repo(demo, root)), demo.base_commit) as env:
        res = run_agent(policy_factory(), env, demo.problem_statement, demo.test_cmd, max_steps)
    return grade(demo, res.patch, root).resolved


# ---- repo-context perturbation -----------------------------------------------------------------
_DISTRACTORS = [
    ("text_helpers", '''"""Small text helpers kept for backwards compatibility."""


def collapse_whitespace(value):
    """Collapse runs of whitespace into single spaces."""
    return " ".join(str(value).split())


def truncate(value, width=80, suffix="..."):
    """Shorten `value` to at most `width` characters."""
    value = str(value)
    if len(value) <= width:
        return value
    return value[: max(0, width - len(suffix))] + suffix
'''),
    ("cache_utils", '''"""Tiny memoisation helpers (unused by the core code paths)."""
import functools


def memoize(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]

    wrapper.cache = cache
    return wrapper
'''),
    ("validators", '''"""Generic input validators."""


def require_positive(name, value):
    if value is None or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def require_non_empty(name, value):
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value
'''),
    ("iter_tools", '''"""Iteration helpers."""


def chunked(iterable, size):
    """Yield lists of at most `size` items."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def first(iterable, default=None):
    for item in iterable:
        return item
    return default
'''),
    ("env_config", '''"""Read optional settings from the environment."""
import os


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
'''),
    ("timing", '''"""Lightweight timing helpers for ad-hoc profiling."""
import time
from contextlib import contextmanager


@contextmanager
def stopwatch(label, sink=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        (sink or (lambda *_: None))(label, elapsed)
'''),
]


def perturb_repo(demo: Demo, out_dir: str | Path, seed: int = 0, *, root: str | Path = ".",
                 rename: bool = True, validate: bool = True, exclude_ids: set[str] = frozenset()) -> Demo | None:
    """Copy demo's repo, add 1-3 unused distractor modules and maybe rename one unreferenced file.

    Returns the new Demo, or None if validation fails (the copied repo is then removed).
    """
    _check_excluded(demo, exclude_ids)
    rng = random.Random(f"perturb:{seed}:{demo.instance_id}")
    new_id = f"{demo.instance_id}-pert{seed}"
    src = _resolve_repo(demo, root)
    dest = Path(out_dir) / new_id
    if dest.exists():
        shutil.rmtree(dest)
    _copy_tree(src, dest)
    files = _read_tree(dest)
    pkgs = sorted({str(Path(f).parent.as_posix()) for f in files
                   if f.endswith("__init__.py") and not is_test_file(f)}) or ["."]
    added = []
    for name, body in rng.sample(_DISTRACTORS, rng.randint(1, 3)):
        pkg = rng.choice(pkgs)
        rel = (Path(pkg) / f"{name}.py").as_posix() if pkg != "." else f"{name}.py"
        if rel in files:
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(body)
        added.append(rel)
    renamed = None
    if rename and rng.random() < 0.5:
        haystack = "\n".join(_read_tree(dest).values()) + demo.patch + demo.test_patch
        touched = set(re.findall(r"^\+\+\+ b/(\S+)", demo.patch + demo.test_patch, re.M))
        cands = [f for f in sorted(files) if f.endswith(".py") and not is_test_file(f)
                 and not f.endswith("__init__.py") and f not in touched
                 and len(re.findall(rf"\b{re.escape(Path(f).stem)}\b", haystack)) == 0]
        if cands:
            old = rng.choice(cands)
            new = (Path(old).parent / f"{Path(old).stem}_legacy.py").as_posix()
            if new not in files:
                (dest / old).rename(dest / new)
                renamed = [old, new]
    d = replace(demo, instance_id=new_id, repo=_repo_field(dest, root), trajectory=[],
                metadata={**demo.metadata, "source": "synthetic:perturb", "parent_id": demo.instance_id,
                          "perturbation": {"added": added, "renamed": renamed}, "num_steps": 0})
    if validate and validate_instance(d, root):
        shutil.rmtree(dest, ignore_errors=True)
        return None
    return d


# ---- driver ------------------------------------------------------------------------------------
def augment(demos: Iterable[Demo], root: str | Path, out_repos_dir: str | Path, max_per_demo: int = 3,
            ops: Iterable[str] = ("mutation",), llm: Callable[[str, str], str] | None = None, seed: int = 0,
            exclude_ids: set[str] = frozenset(), operators: Iterable[str] = OPERATORS,
            max_tries: int | None = None) -> tuple[list[Demo], AugmentStats]:
    """Run the requested augmentation kinds over `demos`; returns (new demos, stats)."""
    ops = list(ops)
    bad = set(ops) - set(KINDS)
    if bad:
        raise ValueError(f"unknown augmentation kinds {sorted(bad)}; choose from {KINDS}")
    demos = list(demos)
    for d in demos:  # refuse up-front, before doing any work
        _check_excluded(d, exclude_ids)
    stats = AugmentStats()
    out: list[Demo] = []
    Path(out_repos_dir).mkdir(parents=True, exist_ok=True)
    for d in demos:
        if "mutation" in ops:
            out += mutate_demo(d, root, out_repos_dir, max_mutants=max_per_demo, operators=operators, llm=llm,
                               seed=seed, exclude_ids=exclude_ids, max_tries=max_tries, stats=stats)
        if "rephrase" in ops:
            if llm is None:
                stats.dropped["rephrase_no_llm"] += 1
            else:
                styles = list(STYLES)
                random.Random(f"reph:{seed}:{d.instance_id}").shuffle(styles)
                for style in styles[:max_per_demo]:
                    try:
                        out.append(rephrase_issue(d, llm, style))
                        stats.kept[f"rephrase:{style}"] += 1
                    except Exception as e:  # noqa: BLE001
                        stats.dropped["rephrase_rejected"] += 1
                        print(f"[augment] {e}", file=sys.stderr)
        if "perturb" in ops:
            p = perturb_repo(d, out_repos_dir, seed, root=root)
            if p is None:
                stats.dropped["perturb_validate_failed"] += 1
            else:
                out.append(p)
                stats.kept["perturb"] += 1
    return out, stats


def cmd_augment(a) -> int:
    from ..schema import load_demos, save_demos

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    llm = None
    if a.issue_model:
        from ..llm import make_text_llm
        llm = make_text_llm(a.issue_model)
    exclude = load_exclude_ids(a.exclude_ids) if a.exclude_ids else set()
    demos = load_demos(a.demos)
    try:
        new, stats = augment(demos, a.root, a.repos_dir, a.max_per_demo, kinds, llm, a.seed, exclude,
                             max_tries=a.max_tries)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    save_demos(new, a.out)
    Path(a.out + ".stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"wrote {len(new)} synthetic instance(s) to {a.out}; repos under {a.repos_dir}.\n"
          f"repo paths are relative to --root ({a.root}) (absolute if the repos dir is outside it): "
          f"validate with `repo-fix validate {a.out} --root {a.root}`")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("augment", help="§3h synthetic instances: bug injection, rephrasing, perturbation")
    p.add_argument("demos", help="input demos JSONL (seeds)")
    p.add_argument("--out", required=True, help="output JSONL of synthetic instances")
    p.add_argument("--repos-dir", default="runs/repos_synth", help="where synthetic repos are written")
    p.add_argument("--max-per-demo", type=int, default=3)
    p.add_argument("--max-tries", type=int, default=None, help="mutants tried per seed (default 8*max-per-demo)")
    p.add_argument("--kinds", default="mutation", help=f"comma-separated subset of {','.join(KINDS)}")
    p.add_argument("--issue-model", default=None, help="model for issue writing/rephrasing (default: offline template)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=".", help="dir that input demo.repo paths (and output ones) are relative to")
    p.add_argument("--exclude-ids", default=None, help="frozen-test manifest; listed ids may not be used as seeds")
    p.set_defaults(fn=cmd_augment)
