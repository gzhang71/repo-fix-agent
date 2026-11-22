import pytest

from accounts import Registry, UsernameTaken


def test_case_insensitive():
    r = Registry()
    r.register("Alice")
    with pytest.raises(UsernameTaken):
        r.register("  alice ")


def test_blank_rejected():
    with pytest.raises(ValueError):
        Registry().register("   ")


def test_is_taken():
    r = Registry()
    r.register("bob")
    assert r.is_taken("BOB") and not r.is_taken("rob")
