"""Helpers for writing seed instance specs. See scripts/seed_specs/README.md."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from repo_fix_agent.schema import Action  # noqa: E402

TEST_CMD = "python -m pytest -q -p no:cacheprovider"


def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def B(cmd):  # bash
    return Action("bash", {"command": cmd})


def V(path, s=None, e=None):  # view_file
    args = {"path": path}
    if s is not None:
        args |= {"start_line": s, "end_line": e}
    return Action("view_file", args)


def R(path, old, new):  # str_replace
    return Action("str_replace", {"path": path, "old_str": old, "new_str": new})


def C(path, content):  # create_file
    return Action("create_file", {"path": path, "content": content})


def S(pattern, path=".", glob=None):  # search
    args = {"pattern": pattern, "path": path}
    if glob:
        args["glob"] = glob
    return Action("search", args)


def F(pattern):  # find_files
    return Action("find_files", {"pattern": pattern})


def I(path, line, text):  # insert
    return Action("insert", {"path": path, "line": line, "text": text})


def T(target=None):  # run_tests
    return Action("run_tests", {"target": target} if target else {})


SUBMIT = Action("submit", {})
