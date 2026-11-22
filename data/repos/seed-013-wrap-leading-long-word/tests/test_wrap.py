import pytest

from textfmt import fill, wrap


def test_basic_wrap():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]


def test_long_word_in_middle():
    assert wrap("a bbbbbbbbbbbb c", 5) == ["a", "bbbbbbbbbbbb", "c"]


def test_empty():
    assert wrap("   ", 10) == []


def test_fill_indent():
    assert fill("one two three", 9, indent="# ") == "# one two\n# three"
    with pytest.raises(ValueError):
        fill("x", 2, indent="-->")
