import pytest

from tags import Tag
from tags.collection import parse_tags


def test_equality_ignores_case():
    assert Tag("Python") == Tag("python")
    assert Tag("go") != Tag("rust")


def test_parse_skips_blanks_and_exact_dupes():
    assert [t.name for t in parse_tags("a, b,,a")] == ["a", "b"]


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        Tag("   ")
