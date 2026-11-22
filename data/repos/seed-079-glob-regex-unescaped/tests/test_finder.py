from srcfind.finder import select
from srcfind.patterns import glob_to_regex


def test_double_star():
    rx = glob_to_regex("**/*.py")
    assert rx.match("a/b/c.py") and rx.match("c.py")
    assert not rx.match("c.pyc")


def test_question_mark():
    assert glob_to_regex("v?/x").match("v1/x")
    assert not glob_to_regex("v?/x").match("v10/x")


def test_select_with_exclude():
    paths = ["src/a.py", "src/b.py", "tests/t.py", "README.md"]
    assert select(paths, ["**/*.py"], ["tests/*"]) == ["src/a.py", "src/b.py"]
