from syncer.paths import is_inside
from syncer.plan import plan_upload


def test_ignores_unrelated_dirs():
    assert plan_upload("/data/proj", ["/data/proj/b/c.txt", "/data/other/x", "/data/proj/a.txt"]) == [
        "a.txt", "b/c.txt"]


def test_is_inside_nested():
    assert is_inside("/data/proj", "/data/proj/deep/er/file")
    assert not is_inside("/data/proj", "/etc/passwd")
