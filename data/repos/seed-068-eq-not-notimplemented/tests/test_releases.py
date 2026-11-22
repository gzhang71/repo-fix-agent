from releases.changelog import Changelog
from releases.version import Version


def test_ordering():
    assert Version("1.2") < Version("1.10")
    assert Version("2.0") >= Version("1.9")
    assert sorted([Version("1.10"), Version("1.2")]) == [Version("1.2"), Version("1.10")]


def test_release_moves_notes():
    cl = Changelog()
    cl.note("fix crash")
    cl.add_release("1.0")
    assert cl.notes[Version("1.0")] == ["fix crash"]
    assert cl.notes["Unreleased"] == []
