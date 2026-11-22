"""Tool surface shared by every policy (engineer demos, teacher, student).

Keeping one small, fixed action space is what makes the 10k engineer demos,
teacher rollouts and student rollouts interchangeable as training data.

`TOOLS` holds Anthropic-style specs; `execute_tool` validates arguments against a
spec and runs the handler against a `Sandbox` (see sandbox.py). It never raises:
every failure becomes an observation starting with "Error:".
"""
from __future__ import annotations

import difflib
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # sandbox.py is optional at import time
    from .sandbox import Sandbox

BASH_TIMEOUT = 120
TEST_TIMEOUT = 300
MAX_OUTPUT = 8000
MAX_TEST_OUTPUT = 4000
VIEW_CAP = 400
SEARCH_CAP = 100
FIND_CAP = 200


@dataclass
class ToolState:
    """Per-episode mutable state, created by the env."""
    edit_history: list[tuple[str, str | None]] = field(default_factory=list)  # (path, previous content | None if new)
    submitted: bool = False
    steps: int = 0


@dataclass
class ToolOutput:
    text: str
    is_error: bool = False
    done: bool = False


class ToolError(Exception):
    """Raised by handlers; becomes ToolOutput(is_error=True, text="Error: ...")."""


# ---- helpers --------------------------------------------------------------------------------
def _norm(path: str) -> str:
    p = posixpath.normpath(path.strip() or ".")
    if p.startswith("/") or p == ".." or p.startswith("../"):
        raise ToolError(f"path must be relative to the repository root: {path}")
    return p


def _check_writable(path: str) -> str:
    p = _norm(path)
    if ".git" in p.split("/"):
        raise ToolError(f"refusing to edit inside .git/: {path}")
    return p


def _read(sb: "Sandbox", path: str) -> str:
    try:
        return sb.read_file(path)
    except FileNotFoundError:
        raise ToolError(f"file not found: {path}") from None


def _read_or_none(sb: "Sandbox", path: str) -> str | None:
    try:
        return sb.read_file(path)
    except FileNotFoundError:
        return None


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _with_exit(out: str, rc: int) -> str:
    return f"{out}\n[exit code {rc}]".lstrip("\n")


def _glob_to_regex(pattern: str) -> re.Pattern:
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?"); i += 3
        elif pattern.startswith("**", i):
            out.append(".*"); i += 2
        elif pattern[i] == "*":
            out.append("[^/]*"); i += 1
        elif pattern[i] == "?":
            out.append("[^/]"); i += 1
        else:
            out.append(re.escape(pattern[i])); i += 1
    return re.compile("".join(out) + r"\Z")


# ---- handlers -------------------------------------------------------------------------------
def _bash(sb, state, command: str) -> ToolOutput:
    r = sb.exec(command, timeout=BASH_TIMEOUT)
    if r.timed_out:
        return ToolOutput(f"Error: command timed out after {BASH_TIMEOUT}s\n{_tail(r.output, 2000)}".rstrip(),
                          is_error=True)
    return ToolOutput(_with_exit(_tail(r.output, MAX_OUTPUT), r.returncode))


def _list_dir(sb, path: str) -> str:
    r = sb.exec(f"ls -1Ap -- {shlex.quote(path)}", timeout=30)
    if r.returncode != 0:
        raise ToolError(r.output.strip() or f"cannot list {path}")
    entries = [e for e in r.output.splitlines() if e and e != ".git/"]
    return f"Directory {path}/:\n" + ("\n".join(entries) if entries else "(empty)")


def _view_file(sb, state, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    p = _norm(path)
    try:
        text = sb.read_file(p)
    except FileNotFoundError:
        raise ToolError(f"file not found: {path}") from None
    except (IsADirectoryError, PermissionError, OSError):
        if sb.exec(f"test -d {shlex.quote(p)}", timeout=10).returncode == 0:
            return _list_dir(sb, p)
        raise
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return f"[{path} is empty]"
    s = max(1, start_line or 1)
    if s > n:
        raise ToolError(f"start_line {s} is past the end of {path} ({n} lines).")
    capped = end_line is None and n - s + 1 > VIEW_CAP
    e = s + VIEW_CAP - 1 if capped else min(n, end_line or n)
    if e < s:
        raise ToolError(f"end_line {e} is before start_line {s}.")
    body = "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(s, e + 1))
    if capped:
        body += f"\n[showing lines {s}-{e} of {n}; pass start_line/end_line to see more]"
    return body


def _closest_hint(text: str, old_str: str) -> str:
    probe = next((l.strip() for l in old_str.splitlines() if l.strip()), "")
    if not probe:
        return ""
    lines = text.splitlines()
    stripped = [l.strip() for l in lines]
    best = difflib.get_close_matches(probe, stripped, n=1, cutoff=0.5)
    if not best:
        return ""
    i = stripped.index(best[0])
    return f" Closest line {i + 1}: {lines[i]!r}. Check whitespace and view the file again."


def _str_replace(sb, state, path: str, old_str: str, new_str: str) -> str:
    p = _check_writable(path)
    text = _read(sb, p)
    n = text.count(old_str) if old_str else 0
    if n != 1:
        hint = _closest_hint(text, old_str) if n == 0 else ""
        raise ToolError(f"old_str occurs {n} times in {path}; it must occur exactly once.{hint}")
    sb.write_file(p, text.replace(old_str, new_str, 1))
    state.edit_history.append((p, text))
    return f"Edited {path}."


def _create_file(sb, state, path: str, content: str) -> str:
    p = _check_writable(path)
    prev = _read_or_none(sb, p)
    sb.write_file(p, content)
    state.edit_history.append((p, prev))
    return f"Wrote {path} ({len(content.splitlines())} lines)."


def _insert(sb, state, path: str, line: int, text: str) -> str:
    p = _check_writable(path)
    content = _read(sb, p)
    lines = content.splitlines(keepends=True)
    if not 0 <= line <= len(lines):
        raise ToolError(f"line must be between 0 and {len(lines)} for {path}.")
    if lines and line == len(lines) and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    new = text if text.endswith("\n") else text + "\n"
    lines.insert(line, new)
    sb.write_file(p, "".join(lines))
    state.edit_history.append((p, content))
    return f"Inserted {len(new.splitlines())} lines after line {line} in {path}."


def _undo_edit(sb, state, path: str) -> str:
    p = _norm(path)
    for idx in range(len(state.edit_history) - 1, -1, -1):
        if state.edit_history[idx][0] == p:
            _, prev = state.edit_history.pop(idx)
            break
    else:
        raise ToolError(f"no edits to undo for {path}.")
    if prev is None:
        r = sb.exec(f"rm -f -- {shlex.quote(p)}", timeout=30)
        if r.returncode != 0:
            raise ToolError(f"could not remove {path}: {r.output.strip()}")
        return f"Undid last edit to {path} (file removed)."
    sb.write_file(p, prev)
    return f"Undid last edit to {path}."


def _search(sb, state, pattern: str, path: str = ".", glob: str | None = None) -> str:
    p = _norm(path)
    cmd = "grep -rnIE --exclude-dir=.git"
    if glob:
        cmd += f" --include={shlex.quote(glob)}"
    cmd += f" -e {shlex.quote(pattern)} -- {shlex.quote(p)}"
    r = sb.exec(cmd, timeout=60)
    if r.returncode == 1:
        return "No matches."
    if r.returncode != 0:
        raise ToolError(f"search failed: {r.output.strip()}")
    hits = [h[2:] if h.startswith("./") else h for h in r.output.splitlines() if h]
    hits = [h if len(h) <= 300 else h[:300] + "..." for h in hits]
    out = "\n".join(hits[:SEARCH_CAP])
    if len(hits) > SEARCH_CAP:
        out += f"\n... {len(hits) - SEARCH_CAP} more matches (narrow the pattern, path or glob)"
    return out


def _find_files(sb, state, pattern: str) -> str:
    cmd = ("git ls-files -co --exclude-standard 2>/dev/null || "
           "find . -type f -not -path './.git/*' | sed 's|^\\./||'")
    r = sb.exec(cmd, timeout=60)
    pat = pattern[2:] if pattern.startswith("./") else pattern
    rx = _glob_to_regex(pat)
    files = sorted({f for f in r.output.splitlines() if f})
    if "/" in pat:
        hits = [f for f in files if rx.match(f)]
    else:  # bare pattern like "test_*.py" matches basenames anywhere
        hits = [f for f in files if rx.match(posixpath.basename(f))]
    if not hits:
        return f"No files match {pattern!r}."
    out = "\n".join(hits[:FIND_CAP])
    if len(hits) > FIND_CAP:
        out += f"\n... {len(hits) - FIND_CAP} more files"
    return out


def _run_tests(sb, state, target: str | None = None) -> ToolOutput:
    cmd = "python -m pytest -q -p no:cacheprovider"
    if target:
        cmd += " " + " ".join(shlex.quote(t) for t in shlex.split(target))
    r = sb.exec(cmd, timeout=TEST_TIMEOUT)
    if r.timed_out:
        return ToolOutput(f"Error: tests timed out after {TEST_TIMEOUT}s\n{_tail(r.output, 2000)}".rstrip(),
                          is_error=True)
    return ToolOutput(_with_exit(_tail(r.output, MAX_TEST_OUTPUT), r.returncode))


def _submit(sb, state) -> ToolOutput:
    state.submitted = True
    return ToolOutput("Submitted.", done=True)


# ---- registry -------------------------------------------------------------------------------
def _spec(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required,
                             "additionalProperties": False}}


_S = {"type": "string"}
_I = {"type": "integer"}
_PATH = {"type": "string", "description": "Path relative to the repo root."}

_REGISTRY: dict[str, tuple[dict, Callable]] = {
    "bash": (_spec("bash", (
        "Run a shell command from the repo root, non-interactively (no stdin), with a "
        f"{BASH_TIMEOUT}s timeout. Output is the last {MAX_OUTPUT} chars plus the exit code. "
        "Do not edit files with it; use str_replace, insert or create_file."),
        {"command": {"type": "string", "description": "Shell command to run."}}, ["command"]), _bash),
    "view_file": (_spec("view_file", (
        "Show a file with line numbers, optionally a 1-indexed inclusive line range. Without "
        f"end_line at most {VIEW_CAP} lines are shown. On a directory, lists its entries."),
        {"path": _PATH, "start_line": _I, "end_line": _I}, ["path"]), _view_file),
    "str_replace": (_spec("str_replace", (
        "Replace one exact occurrence of old_str with new_str in a file. old_str must appear "
        "exactly once, including whitespace; add surrounding lines to make it unique."),
        {"path": _PATH, "old_str": _S, "new_str": _S}, ["path", "old_str", "new_str"]), _str_replace),
    "create_file": (_spec("create_file", "Create or overwrite a file with the given content.",
                          {"path": _PATH, "content": _S}, ["path", "content"]), _create_file),
    "search": (_spec("search", (
        "Regex search (grep -E syntax) over file contents, excluding .git. Returns "
        f"path:line:text, at most {SEARCH_CAP} matches."),
        {"pattern": {"type": "string", "description": "Extended regular expression."},
         "path": {"type": "string", "description": "File or directory to search. Default: repo root."},
         "glob": {"type": "string", "description": "Only search files whose name matches, e.g. '*.py'."}},
        ["pattern"]), _search),
    "find_files": (_spec("find_files", (
        "Find files by glob, e.g. '**/test_*.py' or 'src/*/models.py'. A pattern without '/' "
        f"matches file names anywhere. Sorted, at most {FIND_CAP} results."),
        {"pattern": _S}, ["pattern"]), _find_files),
    "insert": (_spec("insert", (
        "Insert text after the given 1-indexed line of an existing file (line 0 = top of file)."),
        {"path": _PATH, "line": _I, "text": _S}, ["path", "line", "text"]), _insert),
    "undo_edit": (_spec("undo_edit", (
        "Revert the most recent str_replace, insert or create_file on a path. Repeat to undo further."),
        {"path": _PATH}, ["path"]), _undo_edit),
    "run_tests": (_spec("run_tests", (
        "Run the repo's tests with pytest (`python -m pytest -q`). Optional target, e.g. "
        f"'tests/test_x.py::test_y'. Returns the last {MAX_TEST_OUTPUT} chars plus the exit code."),
        {"target": {"type": "string", "description": "pytest node id, file or directory."}}, []), _run_tests),
    "submit": (_spec("submit", (
        "Finish the task. Call this once you have fixed the bug and verified it. "
        "The current diff is your answer."), {}, []), _submit),
}

TOOLS: list[dict] = [spec for spec, _ in _REGISTRY.values()]
TOOL_NAMES: list[str] = [t["name"] for t in TOOLS]

_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
          "object": (dict,), "array": (list,)}


def _validate(schema: dict, args) -> None:
    if not isinstance(args, dict):
        raise ToolError(f"arguments must be a JSON object, got {type(args).__name__}")
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}; allowed: {', '.join(props) or 'none'}")
    for k, v in args.items():
        t = props[k].get("type")
        ok = isinstance(v, _TYPES.get(t, (object,)))
        if t != "boolean" and isinstance(v, bool):
            ok = False
        if not ok:
            raise ToolError(f"argument {k!r} must be of type {t}, got {type(v).__name__}")


def execute_tool(sandbox: "Sandbox", state: ToolState, name: str, args: dict) -> ToolOutput:
    """Validate and run one tool call. Never raises."""
    state.steps += 1
    entry = _REGISTRY.get(name)
    if entry is None:
        return ToolOutput(f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}", is_error=True)
    spec, handler = entry
    try:
        _validate(spec["input_schema"], args)
        result = handler(sandbox, state, **args)
    except ToolError as e:
        return ToolOutput(f"Error: {e}", is_error=True)
    except Exception as e:  # tool failures are observations, not crashes
        return ToolOutput(f"Error: {type(e).__name__}: {e}", is_error=True)
    return result if isinstance(result, ToolOutput) else ToolOutput(result)
