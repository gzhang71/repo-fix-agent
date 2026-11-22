"""§3d sub-skill auxiliary tasks mined from demos (single-turn chat examples with automatic labels).

Every example is `{"id", "task", "messages": [system, user, assistant], "meta": {...}}`.

  localization  issue + file tree            -> touched files, then enclosing defs   (score_localization)
  patchgen      issue + gold-touched source  -> str_replace JSON calls               (verify_patchgen)
  testwrite     issue + relevant source      -> hidden test file                     (verify_test)
  repro         issue                        -> script: exit!=0 before, 0 after      (verify_repro)
  critique      issue + candidate patch      -> RESOLVED / UNRESOLVED + reason       (score_critique)
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..env import RepoEnv
from ..grader import GradeResult, grade
from ..schema import Action, Demo, load_demos
from .sft_data import (check_excluded, guard_frozen, instance_key, load_exclude_ids, patch_files, patches_equivalent,
                       repo_path, write_jsonl)

ALL_TASKS = ("localization", "patchgen", "testwrite", "repro", "critique")
_SKIP_DIRS = {".git", "tests", "test", "__pycache__", ".pytest_cache", ".venv", "node_modules", ".tox"}


# ---- unified-diff parsing ---------------------------------------------------------------------
@dataclass
class Hunk:
    old_start: int
    old_len: int
    lines: list[str]  # raw hunk body lines (" ", "-", "+" prefixed)


@dataclass
class FileDiff:
    old_path: str | None   # None for new files
    new_path: str | None   # None for deleted files
    hunks: list[Hunk]

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    old: str | None = None
    for ln in patch.splitlines():
        if ln.startswith("diff --git "):
            cur, old = None, None
        elif ln.startswith("--- ") and cur is None:
            p = ln[4:].strip()
            old = None if p == "/dev/null" else (p[2:] if p.startswith("a/") else p)
        elif ln.startswith("+++ ") and cur is None:
            p = ln[4:].strip()
            new = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
            cur = FileDiff(old, new, [])
            files.append(cur)
        elif cur is not None and (m := _HUNK_RE.match(ln)):
            cur.hunks.append(Hunk(int(m.group(1)), int(m.group(2) or 1), []))
        elif cur is not None and cur.hunks and ln[:1] in (" ", "-", "+", ""):
            cur.hunks[-1].lines.append(ln if ln else " ")
    return files


def apply_file_diff(original: str, fd: FileDiff) -> str:
    """Apply one file's hunks to its original text (pure; hunks must match exactly)."""
    src = original.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for h in fd.hunks:
        start = h.old_start - 1 if h.old_len else h.old_start
        out.extend(src[pos:start])
        pos = start
        for ln in h.lines:
            tag, body = ln[0], ln[1:] + "\n"
            if tag == " ":
                out.append(src[pos]); pos += 1
            elif tag == "-":
                pos += 1
            elif tag == "+":
                out.append(body)
    out.extend(src[pos:])
    return "".join(out)


def changed_old_lines(fd: FileDiff) -> list[int]:
    """1-based line numbers in the original file touched by the diff (insertions map to the line above)."""
    lines: set[int] = set()
    for h in fd.hunks:
        n = h.old_start if h.old_len else h.old_start + 1
        for ln in h.lines:
            if ln[0] == " ":
                n += 1
            elif ln[0] == "-":
                lines.add(n); n += 1
            elif ln[0] == "+":
                lines.add(max(1, n - 1))
    return sorted(lines)


def enclosing_symbols(source: str, line_numbers: list[int]) -> list[tuple[str, int, int]]:
    """Innermost def/class (qualified name, start, end) enclosing each line; module-level lines skipped."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans: list[tuple[str, int, int]] = []

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                spans.append((name, start, child.end_lineno or child.lineno))
                walk(child, name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    out: list[tuple[str, int, int]] = []
    for n in line_numbers:
        inner = [s for s in spans if s[1] <= n <= s[2]]
        if inner:
            best = min(inner, key=lambda s: s[2] - s[1])
            if best not in out:
                out.append(best)
    return out


# ---- repo access ------------------------------------------------------------------------------
class RepoReader:
    """Reads files at the demo's base state: directly for local repos, via a RepoEnv for git URLs."""

    def __init__(self, demo: Demo, root: str | Path = "."):
        self.path = repo_path(demo, root)
        self.env = RepoEnv(self.path, demo.base_commit) if "://" in self.path else None

    def read(self, rel: str) -> str | None:
        try:
            if self.env:
                return self.env.sandbox.read_file(rel)
            return Path(self.path, rel).read_text()
        except (FileNotFoundError, IsADirectoryError, ValueError):
            return None

    def tree(self) -> list[str]:
        if self.env:
            files = self.env.run("git ls-files").output.splitlines()
            return sorted(f for f in files if f and not set(f.split("/")[:-1]) & _SKIP_DIRS)
        out = []
        for dirpath, dirnames, filenames in os.walk(self.path):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            rel = os.path.relpath(dirpath, self.path)
            for fn in sorted(filenames):
                if fn.endswith((".pyc", ".pyo")):
                    continue
                out.append(fn if rel == "." else f"{rel}/{fn}".replace(os.sep, "/"))
        return out

    def close(self) -> None:
        if self.env:
            self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def gold_locations(demo: Demo, reader: RepoReader) -> tuple[list[str], list[str]]:
    """(files touched by demo.patch, "path::Qualified.name" of innermost enclosing defs)."""
    files, funcs = [], []
    for fd in parse_patch(demo.patch):
        files.append(fd.path)
        src = reader.read(fd.old_path) if fd.old_path else None
        if src is not None and fd.path.endswith(".py"):
            funcs += [f"{fd.path}::{name}" for name, _, _ in enclosing_symbols(src, changed_old_lines(fd))]
    return files, funcs


def relevant_source(demo: Demo, reader: RepoReader, window: int = 5) -> str:
    """Source of the gold-touched functions (or a line window for module-level edits)."""
    blocks = []
    for fd in parse_patch(demo.patch):
        if not fd.old_path:
            continue
        src = reader.read(fd.old_path)
        if src is None:
            continue
        lines = src.splitlines()
        changed = changed_old_lines(fd)
        syms = enclosing_symbols(src, changed) if fd.path.endswith(".py") else []
        covered = set()
        spans = []
        for name, s, e in syms:
            spans.append((s, e, name))
            covered |= set(range(s, e + 1))
        for n in changed:
            if n not in covered:
                spans.append((max(1, n - window), min(len(lines), n + window), None))
                covered |= set(range(max(1, n - window), n + window + 1))
        for s, e, name in sorted(set(spans)):
            label = f"{fd.old_path}" + (f" ({name}, lines {s}-{e})" if name else f" (lines {s}-{e})")
            blocks.append(f"# {label}\n" + "\n".join(lines[s - 1:e]))
    return "\n\n".join(blocks)


# ---- task: fault localization -----------------------------------------------------------------
LOCALIZATION_SYSTEM = """You localize bugs. Given an issue and the repository's file list, name the source
files that must change to fix it (most likely first), then the functions or classes to change.
Answer exactly in the format:
FILES:
<path>
FUNCTIONS:
<path>::<Qualified.name>"""


def format_localization(files: list[str], funcs: list[str]) -> str:
    return "FILES:\n" + "\n".join(files) + "\nFUNCTIONS:\n" + "\n".join(funcs)


def parse_localization(text: str) -> tuple[list[str], list[str]]:
    files, funcs, cur = [], [], None
    for ln in text.splitlines():
        s = ln.strip().lstrip("-* ").strip("`")
        if s.upper().startswith("FILES:"):
            cur = files
        elif s.upper().startswith("FUNCTIONS:"):
            cur = funcs
        elif s and cur is not None:
            cur.append(s)
    return files, funcs


def score_localization(pred_files: list[str], gold_files: list[str], k: int = 5) -> float:
    """Recall@k: fraction of gold files found among the top-k predictions."""
    gold = set(gold_files)
    if not gold:
        return 1.0
    return len(set(pred_files[:k]) & gold) / len(gold)


def localization_example(demo: Demo, reader: RepoReader) -> dict | None:
    files, funcs = gold_locations(demo, reader)
    if not files:
        return None
    tree = reader.tree()
    user = f"<issue>\n{demo.problem_statement.strip()}\n</issue>\n\nRepository files:\n" + "\n".join(tree)
    return _example(demo, "localization", LOCALIZATION_SYSTEM, user, format_localization(files, funcs),
                    {"gold_files": files, "gold_functions": funcs})


# ---- task: patch generation as str_replace calls ----------------------------------------------
PATCHGEN_SYSTEM = """You fix bugs. Given an issue and the relevant source code, output the edit as a JSON
list of tool calls, each {"tool": "str_replace", "args": {"path": ..., "old_str": ..., "new_str": ...}}
(or "create_file" with "path" and "content" for new files). Apply them in order. Output JSON only."""


def _str_replace_calls(path: str, original: str, patched: str, merge_gap: int = 2) -> list[dict]:
    import difflib

    a, b = original.splitlines(keepends=True), patched.splitlines(keepends=True)
    regions: list[list[int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if regions and i1 - regions[-1][1] <= merge_gap:
            regions[-1][1], regions[-1][3] = i2, j2
        else:
            regions.append([i1, i2, j1, j2])
    state = list(a)
    calls = []
    for i1, i2, j1, j2 in reversed(regions):  # bottom-up keeps earlier indices valid
        lo, hi = i1, i2
        if lo == hi:  # pure insertion: anchor on a neighbouring line
            if lo > 0:
                lo -= 1
            else:
                hi += 1
        text = "".join(state)
        while text.count("".join(state[lo:hi])) != 1 or not "".join(state[lo:hi]).strip():
            if lo == 0 and hi == len(state):
                break
            lo, hi = max(0, lo - 1), min(len(state), hi + 1)
        old = "".join(state[lo:hi])
        new = "".join(state[lo:i1]) + "".join(b[j1:j2]) + "".join(state[i2:hi])
        calls.append({"tool": "str_replace", "args": {"path": path, "old_str": old, "new_str": new}})
        state[lo:hi] = (state[lo:i1] + b[j1:j2] + state[i2:hi])
    return calls


def patch_to_calls(demo: Demo, reader: RepoReader) -> list[dict] | None:
    """Express demo.patch as str_replace / create_file calls; None if not expressible (deletions)."""
    calls = []
    for fd in parse_patch(demo.patch):
        if fd.new_path is None:
            return None
        if fd.old_path is None:
            calls.append({"tool": "create_file", "args": {"path": fd.new_path,
                                                            "content": apply_file_diff("", fd)}})
            continue
        if fd.old_path != fd.new_path:
            return None
        src = reader.read(fd.old_path)
        if src is None:
            return None
        try:
            patched = apply_file_diff(src, fd)
        except IndexError:
            return None
        calls += _str_replace_calls(fd.path, src, patched)
    return calls


def parse_calls(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    data = json.loads(m.group(0) if m else text)
    return [c for c in data if isinstance(c, dict) and "tool" in c and isinstance(c.get("args"), dict)]


def calls_to_patch(demo: Demo, calls: list[dict], root: str | Path = ".", backend: str = "auto") -> str:
    """Apply predicted tool calls in a fresh RepoEnv; return the resulting diff (raises on a failed call)."""
    with RepoEnv(repo_path(demo, root), demo.base_commit, backend=backend) as env:
        for c in calls:
            if c["tool"] not in ("str_replace", "create_file", "insert"):
                raise ValueError(f"unsupported tool in patch answer: {c['tool']}")
            obs = env.execute(Action(c["tool"], c["args"]))
            if obs.startswith("Error:"):
                raise ValueError(obs)
        return env.diff()


def verify_patchgen(demo: Demo, answer: str | list[dict], root: str | Path = ".") -> GradeResult:
    try:
        calls = parse_calls(answer) if isinstance(answer, str) else answer
        patch = calls_to_patch(demo, calls, root)
    except (ValueError, KeyError, TypeError) as e:
        return GradeResult(False, error=f"could not apply answer: {e}")
    return grade(demo, patch, root)


def patchgen_example(demo: Demo, reader: RepoReader) -> dict | None:
    calls = patch_to_calls(demo, reader)
    if not calls:
        return None
    ctx = relevant_source(demo, reader)
    user = f"<issue>\n{demo.problem_statement.strip()}\n</issue>\n\nRelevant code:\n```\n{ctx}\n```"
    return _example(demo, "patchgen", PATCHGEN_SYSTEM, user, json.dumps(calls, indent=1),
                    {"files": patch_files(demo.patch)})


# ---- task: test writing -----------------------------------------------------------------------
TESTWRITE_SYSTEM = """You write regression tests. Given an issue and the relevant source code, write a
pytest test file that fails on the current (buggy) code and passes once the bug is fixed.
Start with a line `# file: <path>` and then the file content."""


@dataclass
class ExecVerdict:
    before_rc: int
    after_rc: int
    ok: bool
    output: str = ""


def hidden_test_files(demo: Demo, reader: RepoReader | None = None) -> dict[str, str]:
    """Post-test_patch content of each test file holding a FAIL_TO_PASS test."""
    f2p_files = {t.split("::")[0] for t in demo.FAIL_TO_PASS}
    out = {}
    for fd in parse_patch(demo.test_patch):
        if fd.new_path is None or (f2p_files and fd.new_path not in f2p_files):
            continue
        base = "" if fd.old_path is None else (reader.read(fd.old_path) if reader else None)
        if base is None:
            continue
        out[fd.new_path] = apply_file_diff(base, fd)
    return out


def _split_file_answer(text: str, default: str) -> tuple[str, str]:
    m = re.match(r"\s*#\s*file:\s*(\S+)\s*\n", text)
    return (m.group(1), text[m.end():]) if m else (default, text)


def _run_both_ways(demo: Demo, path: str, content: str, cmd: str, root, backend: str = "auto") -> ExecVerdict:
    rcs, outs = [], []
    for patch in ("", demo.patch):
        with RepoEnv(repo_path(demo, root), demo.base_commit, backend=backend) as env:
            env.apply_patch(patch)
            env.sandbox.write_file(path, content)
            r = env.run(cmd, timeout=300)
            rcs.append(r.returncode)
            outs.append(r.output[-1500:])
    return ExecVerdict(rcs[0], rcs[1], rcs[0] != 0 and rcs[1] == 0, "\n---\n".join(outs))


def verify_test(demo: Demo, test_code: str, root: str | Path = ".", path: str | None = None,
                backend: str = "auto") -> ExecVerdict:
    """Candidate test must fail on the buggy repo and pass with the gold patch applied."""
    default = path or f"tests/test_aux_candidate_{re.sub(r'[^A-Za-z0-9_]', '_', demo.instance_id)}.py"
    p, code = _split_file_answer(test_code, default)
    if path:
        p = path
    return _run_both_ways(demo, p, code, f"{demo.test_cmd} {shlex.quote(p)}", root, backend)


def testwrite_example(demo: Demo, reader: RepoReader, root, verify: bool = True) -> dict | None:
    files = hidden_test_files(demo, reader)
    if len(files) != 1:
        return None
    path, content = next(iter(files.items()))
    target = f"# file: {path}\n{content}"
    if verify and not verify_test(demo, target, root).ok:
        return None
    ctx = relevant_source(demo, reader)
    user = f"<issue>\n{demo.problem_statement.strip()}\n</issue>\n\nRelevant code:\n```\n{ctx}\n```"
    return _example(demo, "testwrite", TESTWRITE_SYSTEM, user, target, {"test_path": path, "verified": verify})


# ---- task: reproduction script ----------------------------------------------------------------
REPRO_SYSTEM = """You reproduce bugs. Given an issue, write a standalone Python script, run from the
repository root, that exits with a non-zero status while the bug is present and with status 0 once it
is fixed. Start with a line `# file: <path>` and then the script."""
REPRO_PATH = "reproduce_issue.py"


def synthesize_repro(demo: Demo, reader: RepoReader | None = None) -> str | None:
    """Turn the F2P tests into a script: test module source + a runner calling each F2P test."""
    files = hidden_test_files(demo, reader)
    if not files:
        return None
    parts = ["import sys", "import traceback", ""]
    calls = []
    for path, content in files.items():
        parts.append(f"# --- from {path}")
        parts.append(content)
        for t in demo.FAIL_TO_PASS:
            if t.split("::")[0] != path:
                continue
            chain = [x.split("[")[0] for x in t.split("::")[1:]]
            if len(chain) == 1:
                calls.append(chain[0])
            elif len(chain) == 2:
                calls.append(f"{chain[0]}().{chain[1]}")
    if not calls:
        return None
    parts += ["", "", "if __name__ == \"__main__\":", "    failed = 0"]
    for c in calls:
        parts += ["    try:", f"        {c}()", "    except Exception:",
                  "        traceback.print_exc()", "        failed += 1"]
    parts += ["    print('BUG REPRODUCED' if failed else 'OK: behaves as expected')",
              "    sys.exit(1 if failed else 0)", ""]
    return "\n".join(parts)


def verify_repro(demo: Demo, script: str, root: str | Path = ".", backend: str = "auto") -> ExecVerdict:
    p, code = _split_file_answer(script, REPRO_PATH)
    return _run_both_ways(demo, p, code, f"python {shlex.quote(p)}", root, backend)


def repro_example(demo: Demo, reader: RepoReader, root) -> dict | None:
    script = synthesize_repro(demo, reader)
    if script is None:
        return None
    target = f"# file: {REPRO_PATH}\n{script}"
    if not verify_repro(demo, target, root).ok:  # only emit examples that verify
        return None
    user = f"<issue>\n{demo.problem_statement.strip()}\n</issue>"
    return _example(demo, "repro", REPRO_SYSTEM, user, target, {"verified": True})


# ---- task: patch critique ---------------------------------------------------------------------
CRITIQUE_SYSTEM = """You review bug fixes. Given an issue and a candidate patch, decide whether the patch
resolves the issue without breaking existing behavior. Answer `RESOLVED` or `UNRESOLVED` on the first
line, followed by a one-sentence reason."""


def _critique_reason(rollout: Demo, g: GradeResult | None) -> str:
    if not rollout.patch.strip():
        return "The candidate makes no changes to the code."
    info = g or rollout.metadata.get("grade")
    if isinstance(info, GradeResult):
        info = {"error": info.error, "fail_to_pass": info.fail_to_pass, "pass_to_pass": info.pass_to_pass}
    if isinstance(info, dict):
        if info.get("error"):
            return "The patch does not apply cleanly to the repository."
        if info.get("pass_to_pass") and not all(info["pass_to_pass"].values()):
            return "The patch breaks previously passing tests (regression)."
        if info.get("fail_to_pass") and not all(info["fail_to_pass"].values()):
            return "The behavior described in the issue is still wrong after the patch."
    return "The patch does not fix the behavior described in the issue."


def critique_examples(demo: Demo, failed: list[Demo], root=".", regrade: bool = False) -> list[dict]:
    files = patch_files(demo.patch)
    gold_reason = f"The patch fixes the root cause in {', '.join(files)} and keeps existing tests passing."
    out = [_critique(demo, "gold", demo.patch, True, gold_reason, demo.metadata.get("source") or "engineer")]
    k = 0
    for r in failed:
        if r.metadata.get("resolved") is not False or patches_equivalent(r.patch, demo.patch):
            continue
        g = grade(demo, r.patch, root) if regrade and r.patch.strip() else None
        if g is not None and g.resolved:
            continue  # label disagrees with execution; skip rather than emit a wrong label
        out.append(_critique(demo, f"neg{k}", r.patch, False, _critique_reason(r, g), r.metadata.get("source")))
        k += 1
    return out


def _critique(demo: Demo, tag: str, patch: str, label: bool, reason: str, source) -> dict:
    user = (f"<issue>\n{demo.problem_statement.strip()}\n</issue>\n\nCandidate patch:\n"
            f"```diff\n{patch.strip() or '(empty patch)'}\n```")
    ex = _example(demo, "critique", CRITIQUE_SYSTEM, user,
                  f"{'RESOLVED' if label else 'UNRESOLVED'}\n{reason}",
                  {"label": label, "candidate_source": source})
    ex["id"] = f"{demo.instance_id}:critique:{tag}"
    return ex


def score_critique(pred_text: str, example: dict) -> bool:
    first = (pred_text.strip().split() or [""])[0].upper().strip(".:*`")
    if first not in ("RESOLVED", "UNRESOLVED"):
        return False
    return (first == "RESOLVED") == bool(example["meta"]["label"])


# ---- driver -----------------------------------------------------------------------------------
def _example(demo: Demo, task: str, system: str, user: str, target: str, meta: dict) -> dict:
    return {"id": f"{demo.instance_id}:{task}", "task": task,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": target}],
            "meta": {"instance_id": demo.instance_id, "repo": demo.repo, **meta}}


def build_aux_tasks(demos: list[Demo], tasks: tuple[str, ...] | list[str] = ALL_TASKS,
                    rollouts: list[Demo] | None = None, root: str | Path = ".",
                    exclude_ids: set[str] | None = None, verify: bool = True,
                    regrade: bool = False) -> list[dict]:
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise ValueError(f"unknown aux task(s): {sorted(unknown)}")
    rollouts = rollouts or []
    check_excluded(demos, exclude_ids, "demos")
    check_excluded(rollouts, exclude_ids, "rollouts")
    by_parent: dict[str, list[Demo]] = {}
    for r in rollouts:
        by_parent.setdefault(instance_key(r), []).append(r)
    out: list[dict] = []
    for d in demos:
        with RepoReader(d, root) as reader:
            if "localization" in tasks and (ex := localization_example(d, reader)):
                out.append(ex)
            if "patchgen" in tasks and (ex := patchgen_example(d, reader)):
                if not verify or verify_patchgen(d, ex["messages"][2]["content"], root).resolved:
                    out.append(ex)
            if "testwrite" in tasks and (ex := testwrite_example(d, reader, root, verify)):
                out.append(ex)
            if "repro" in tasks and (ex := repro_example(d, reader, root)):
                out.append(ex)
            if "critique" in tasks:
                out += critique_examples(d, by_parent.get(d.instance_id, []), root, regrade)
    return out


def cmd_aux_tasks(a) -> int:
    exclude = load_exclude_ids(a.exclude_ids)
    demos = load_demos(a.demos)
    rollouts = load_demos(a.rollouts) if a.rollouts else []
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    exs = build_aux_tasks(demos, tasks, rollouts, a.root, exclude, verify=not a.no_verify, regrade=a.regrade)
    write_jsonl(exs, a.out)
    counts: dict[str, int] = {}
    for e in exs:
        counts[e["task"]] = counts.get(e["task"], 0) + 1
    print(json.dumps({"n": len(exs), "by_task": counts}))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("aux-tasks", help="§3d: mine single-turn sub-skill examples from demos")
    p.add_argument("demos")
    p.add_argument("--out", default="runs/aux/aux.jsonl")
    p.add_argument("--tasks", default=",".join(ALL_TASKS))
    p.add_argument("--rollouts", default=None, help="failed rollouts (metadata.resolved=False) for critique")
    p.add_argument("--root", default=".")
    p.add_argument("--no-verify", action="store_true", help="skip execution checks for patchgen/testwrite")
    p.add_argument("--regrade", action="store_true", help="re-grade failed rollouts for critique reasons")
    p.add_argument("--exclude-ids", default=None)
    p.set_defaults(fn=guard_frozen(cmd_aux_tasks))
