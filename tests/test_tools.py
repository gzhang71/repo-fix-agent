import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from repo_fix_agent.tools import TOOL_NAMES, TOOLS, ToolOutput, ToolState, execute_tool  # noqa: E402


@dataclass
class ExecResult:
    returncode: int
    output: str
    timed_out: bool = False


class FakeSandbox:
    """Minimal implementation of the Sandbox protocol over a local directory."""

    def __init__(self, root: Path, init_git: bool = True):
        self.root = str(root)
        self.baseline = ""
        bindir = root.parent / f"{root.name}-bin"
        bindir.mkdir(exist_ok=True)
        if not (bindir / "python").exists():
            (bindir / "python").symlink_to(sys.executable)
        self.env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
                    "PYTHONDONTWRITEBYTECODE": "1"}
        if init_git:
            for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                         "--allow-empty", "-m", "baseline"]):
                subprocess.run(cmd, cwd=root, check=True, capture_output=True)
            self.baseline = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                           text=True).stdout.strip()

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                               timeout=timeout, env=self.env, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            return ExecResult(-1, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), True)
        return ExecResult(r.returncode, r.stdout + r.stderr)

    def _resolve(self, path: str) -> Path:
        root = Path(self.root).resolve()
        p = (root / path).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"path escapes repository: {path}")
        return p

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()

    def write_file(self, path: str, content: str) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "pkg" / "__init__.py").write_text("")
    (r / "pkg" / "mod.py").write_text("def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n")
    (r / "tests").mkdir()
    (r / "tests" / "test_mod.py").write_text(
        "from pkg.mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\ndef test_ok():\n    assert True\n")
    return r


@pytest.fixture
def sb(repo):
    return FakeSandbox(repo)


@pytest.fixture
def st():
    return ToolState()


def run(sb, st, name, **args) -> ToolOutput:
    return execute_tool(sb, st, name, args)


# ---- specs ----------------------------------------------------------------------------------
def test_specs_shape_and_order():
    assert TOOL_NAMES[:4] == ["bash", "view_file", "str_replace", "create_file"]
    assert TOOL_NAMES[-1] == "submit"
    assert set(TOOL_NAMES) == {"bash", "view_file", "str_replace", "create_file", "search", "find_files",
                               "insert", "undo_edit", "run_tests", "submit"}
    for t in TOOLS:
        assert set(t) == {"name", "description", "input_schema"}
        s = t["input_schema"]
        assert s["type"] == "object"
        assert s["additionalProperties"] is False
        assert isinstance(s["required"], list)
        assert set(s["required"]) <= set(s["properties"])
        assert t["description"] and len(t["description"]) < 400


def test_legacy_arg_names_unchanged():
    props = {t["name"]: set(t["input_schema"]["properties"]) for t in TOOLS}
    assert props["bash"] == {"command"}
    assert props["view_file"] == {"path", "start_line", "end_line"}
    assert props["str_replace"] == {"path", "old_str", "new_str"}
    assert props["create_file"] == {"path", "content"}
    assert props["submit"] == set()


# ---- validation -----------------------------------------------------------------------------
def test_unknown_tool(sb, st):
    out = run(sb, st, "rm_rf")
    assert out.is_error and out.text.startswith("Error: unknown tool 'rm_rf'")


@pytest.mark.parametrize("name,args,needle", [
    ("view_file", {}, "missing required"),
    ("bash", {"command": "ls", "cwd": "/"}, "unknown argument"),
    ("bash", {"command": 3}, "must be of type string"),
    ("view_file", {"path": "pkg/mod.py", "start_line": "1"}, "must be of type integer"),
    ("view_file", {"path": "pkg/mod.py", "start_line": True}, "must be of type integer"),
    ("insert", {"path": "pkg/mod.py", "line": False, "text": "x"}, "must be of type integer"),
    ("submit", {"force": True}, "unknown argument"),
])
def test_arg_validation(sb, st, repo, name, args, needle):
    before = (repo / "pkg" / "mod.py").read_text()
    out = execute_tool(sb, st, name, args)
    assert out.is_error and out.text.startswith("Error:") and needle in out.text
    assert (repo / "pkg" / "mod.py").read_text() == before


def test_non_dict_args(sb, st):
    out = execute_tool(sb, st, "bash", ["ls"])
    assert out.is_error


def test_steps_counted(sb, st):
    run(sb, st, "bash", command="true")
    run(sb, st, "nope")
    assert st.steps == 2


def test_handler_exception_becomes_error(st):
    class Broken:
        def read_file(self, path):
            raise RuntimeError("boom")
    out = execute_tool(Broken(), st, "view_file", {"path": "x"})
    assert out.is_error and "boom" in out.text


# ---- bash -----------------------------------------------------------------------------------
def test_bash_format(sb, st):
    out = run(sb, st, "bash", command="echo hi")
    assert out.text == "hi\n\n[exit code 0]" and not out.is_error
    out = run(sb, st, "bash", command="exit 3")
    assert out.text == "[exit code 3]" and not out.is_error


def test_bash_truncates(sb, st):
    out = run(sb, st, "bash", command="python -c 'print(\"x\" * 20000)'")
    assert out.text.startswith("...[truncated]...") and out.text.endswith("[exit code 0]")
    assert len(out.text) < 8100


def test_bash_timeout(st):
    class Slow:
        def exec(self, command, timeout=120):
            assert timeout == 120
            return ExecResult(-1, "partial", timed_out=True)
    out = execute_tool(Slow(), st, "bash", {"command": "sleep 999"})
    assert out.is_error and "timed out" in out.text


# ---- view_file ------------------------------------------------------------------------------
def test_view_file_format(sb, st):
    out = run(sb, st, "view_file", path="pkg/mod.py")
    assert out.text.splitlines()[0] == "   1  def add(a, b):"
    assert out.text.splitlines()[1] == "   2      return a - b"
    out = run(sb, st, "view_file", path="pkg/mod.py", start_line=5, end_line=6)
    assert out.text == "   5  def sub(a, b):\n   6      return a - b"


def test_view_file_errors(sb, st):
    assert run(sb, st, "view_file", path="nope.py").text == "Error: file not found: nope.py"
    assert run(sb, st, "view_file", path="pkg/mod.py", start_line=50).is_error
    assert run(sb, st, "view_file", path="../../etc/passwd").is_error
    assert run(sb, st, "view_file", path="/etc/passwd").is_error


def test_view_file_directory(sb, st):
    out = run(sb, st, "view_file", path="pkg")
    assert not out.is_error and "mod.py" in out.text and "__init__.py" in out.text
    out = run(sb, st, "view_file", path=".")
    assert "pkg/" in out.text and "tests/" in out.text and ".git/" not in out.text


def test_view_file_caps(sb, st, repo):
    (repo / "big.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 1001)))
    out = run(sb, st, "view_file", path="big.py")
    lines = out.text.splitlines()
    assert lines[399] == " 400  x400 = 400" and "showing lines 1-400 of 1000" in lines[-1]
    assert len(lines) == 401
    out = run(sb, st, "view_file", path="big.py", start_line=990, end_line=1000)
    assert len(out.text.splitlines()) == 11


# ---- str_replace ----------------------------------------------------------------------------
def test_str_replace_and_undo(sb, st, repo):
    f = repo / "pkg" / "mod.py"
    orig = f.read_text()
    out = run(sb, st, "str_replace", path="pkg/mod.py", old_str="def add(a, b):\n    return a - b",
              new_str="def add(a, b):\n    return a + b")
    assert out.text == "Edited pkg/mod.py." and not out.is_error
    assert "return a + b" in f.read_text()
    assert run(sb, st, "undo_edit", path="pkg/mod.py").text == "Undid last edit to pkg/mod.py."
    assert f.read_text() == orig
    assert run(sb, st, "undo_edit", path="pkg/mod.py").is_error


def test_str_replace_ambiguous(sb, st):
    out = run(sb, st, "str_replace", path="pkg/mod.py", old_str="return a - b", new_str="x")
    assert out.is_error
    assert out.text == "Error: old_str occurs 2 times in pkg/mod.py; it must occur exactly once."


def test_str_replace_zero_matches_hint(sb, st):
    out = run(sb, st, "str_replace", path="pkg/mod.py", old_str="def add(a,b):", new_str="x")
    assert out.is_error
    assert out.text.startswith("Error: old_str occurs 0 times in pkg/mod.py; it must occur exactly once.")
    assert "Closest line 1" in out.text and "def add(a, b):" in out.text


def test_str_replace_errors(sb, st):
    assert run(sb, st, "str_replace", path="missing.py", old_str="a", new_str="b").is_error
    assert run(sb, st, "str_replace", path="pkg/mod.py", old_str="", new_str="b").is_error
    out = run(sb, st, "str_replace", path=".git/config", old_str="a", new_str="b")
    assert out.is_error and ".git" in out.text
    assert st.edit_history == []


# ---- create_file ----------------------------------------------------------------------------
def test_create_file_new_and_undo(sb, st, repo):
    out = run(sb, st, "create_file", path="pkg/new/thing.py", content="a = 1\nb = 2\n")
    assert out.text == "Wrote pkg/new/thing.py (2 lines)."
    assert (repo / "pkg/new/thing.py").read_text() == "a = 1\nb = 2\n"
    assert "file removed" in run(sb, st, "undo_edit", path="pkg/new/thing.py").text
    assert not (repo / "pkg/new/thing.py").exists()


def test_create_file_overwrite_and_undo(sb, st, repo):
    f = repo / "pkg" / "mod.py"
    orig = f.read_text()
    run(sb, st, "create_file", path="pkg/mod.py", content="x = 1\n")
    assert f.read_text() == "x = 1\n"
    run(sb, st, "undo_edit", path="./pkg/mod.py")  # normalized path
    assert f.read_text() == orig


def test_create_file_rejects_git(sb, st):
    out = run(sb, st, "create_file", path="pkg/../.git/hooks/pre-commit", content="x")
    assert out.is_error and ".git" in out.text


# ---- insert ---------------------------------------------------------------------------------
def test_insert_positions_and_undo(sb, st, repo):
    f = repo / "pkg" / "mod.py"
    orig = f.read_text()
    out = run(sb, st, "insert", path="pkg/mod.py", line=0, text="import os")
    assert out.text == "Inserted 1 lines after line 0 in pkg/mod.py."
    assert f.read_text().startswith("import os\ndef add")
    run(sb, st, "insert", path="pkg/mod.py", line=2, text="    # note\n")
    assert f.read_text().splitlines()[2] == "    # note"
    n = len(f.read_text().splitlines())
    run(sb, st, "insert", path="pkg/mod.py", line=n, text="END")
    assert f.read_text().endswith("    return a - b\nEND\n")
    for _ in range(3):
        assert not run(sb, st, "undo_edit", path="pkg/mod.py").is_error
    assert f.read_text() == orig


def test_insert_no_trailing_newline(sb, st, repo):
    (repo / "a.txt").write_text("one\ntwo")
    run(sb, st, "insert", path="a.txt", line=2, text="three")
    assert (repo / "a.txt").read_text() == "one\ntwo\nthree\n"


def test_insert_errors(sb, st):
    assert run(sb, st, "insert", path="pkg/mod.py", line=99, text="x").is_error
    assert run(sb, st, "insert", path="pkg/mod.py", line=-1, text="x").is_error
    assert run(sb, st, "insert", path="nope.py", line=0, text="x").is_error
    assert run(sb, st, "insert", path=".git/HEAD", line=0, text="x").is_error


# ---- undo across files ----------------------------------------------------------------------
def test_undo_is_per_path(sb, st, repo):
    run(sb, st, "create_file", path="a.py", content="a\n")
    run(sb, st, "str_replace", path="pkg/mod.py", old_str="def sub", new_str="def minus")
    run(sb, st, "undo_edit", path="a.py")
    assert not (repo / "a.py").exists()
    assert "def minus" in (repo / "pkg/mod.py").read_text()
    assert run(sb, st, "undo_edit", path="never.py").is_error


# ---- search ---------------------------------------------------------------------------------
def test_search(sb, st):
    out = run(sb, st, "search", pattern=r"def (add|sub)")
    assert not out.is_error
    assert out.text.splitlines() == ["pkg/mod.py:1:def add(a, b):", "pkg/mod.py:5:def sub(a, b):"]
    out = run(sb, st, "search", pattern="add", path="tests", glob="*.py")
    assert all(line.startswith("tests/") for line in out.text.splitlines())
    assert run(sb, st, "search", pattern="zzz_nothing").text == "No matches."
    assert ".git" not in run(sb, st, "search", pattern="baseline|HEAD|ref").text


def test_search_quoting_and_errors(sb, st, repo):
    (repo / "q.txt").write_text("it's $HOME; `x`\n")
    out = run(sb, st, "search", pattern="it's \\$HOME; `x`")
    assert out.text == "q.txt:1:it's $HOME; `x`"
    assert run(sb, st, "search", pattern="(unclosed").is_error
    assert run(sb, st, "search", pattern="x", path="../").is_error


def test_search_cap(sb, st, repo):
    (repo / "many.txt").write_text("hit\n" * 150)
    out = run(sb, st, "search", pattern="^hit$")
    lines = out.text.splitlines()
    assert len(lines) == 101 and lines[-1].startswith("... 50 more matches")


# ---- find_files -----------------------------------------------------------------------------
def test_find_files(sb, st, repo):
    assert run(sb, st, "find_files", pattern="**/test_*.py").text == "tests/test_mod.py"
    assert run(sb, st, "find_files", pattern="*.py").text.splitlines() == [
        "pkg/__init__.py", "pkg/mod.py", "tests/test_mod.py"]
    assert run(sb, st, "find_files", pattern="pkg/*.py").text.splitlines() == ["pkg/__init__.py", "pkg/mod.py"]
    assert run(sb, st, "find_files", pattern="**/*.rs").text.startswith("No files match")
    (repo / "untracked.py").write_text("")
    assert "untracked.py" in run(sb, st, "find_files", pattern="untracked*").text


def test_find_files_without_git_and_cap(tmp_path, st):
    r = tmp_path / "plain"
    (r / "d").mkdir(parents=True)
    for i in range(250):
        (r / "d" / f"f{i:03d}.txt").write_text("")
    sb = FakeSandbox(r, init_git=False)
    out = run(sb, st, "find_files", pattern="**/*.txt")
    lines = out.text.splitlines()
    assert len(lines) == 201 and lines[0] == "d/f000.txt" and lines[-1] == "... 50 more files"


# ---- run_tests ------------------------------------------------------------------------------
def test_run_tests(sb, st, repo):
    out = run(sb, st, "run_tests")
    assert not out.is_error and "1 failed" in out.text and out.text.endswith("[exit code 1]")
    out = run(sb, st, "run_tests", target="tests/test_mod.py::test_ok")
    assert "1 passed" in out.text and out.text.endswith("[exit code 0]")
    assert not (repo / ".pytest_cache").exists()


def test_run_tests_truncates(st):
    class Loud:
        def exec(self, command, timeout=120):
            assert command == "python -m pytest -q -p no:cacheprovider 'a b.py'"
            return ExecResult(1, "y" * 10000)
    out = execute_tool(Loud(), st, "run_tests", {"target": "'a b.py'"})
    assert out.text.endswith("[exit code 1]") and len(out.text) < 4100


# ---- submit ---------------------------------------------------------------------------------
def test_submit(sb, st):
    out = run(sb, st, "submit")
    assert out == ToolOutput("Submitted.", is_error=False, done=True)
    assert st.submitted
    assert not run(sb, st, "bash", command="true").done


# ---- backward compatibility with recorded demos ---------------------------------------------
def _seed_demos():
    with open(ROOT / "data" / "demos" / "seed.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.parametrize("demo", _seed_demos(), ids=lambda d: d["instance_id"])
def test_replay_seed_demos(tmp_path, demo):
    src = ROOT / demo["repo"]
    dst = tmp_path / "repo"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    sb, st = FakeSandbox(dst), ToolState()
    out = None
    for step in demo["trajectory"]:
        a = step["action"]
        out = execute_tool(sb, st, a["tool"], a["args"])
        assert not out.is_error, (a, out.text)
        if a["tool"] in ("str_replace", "create_file", "submit"):
            assert out.text == step["observation"]
        if a["tool"] == "view_file":
            assert out.text == step["observation"]
    assert out is not None and out.done and st.submitted
