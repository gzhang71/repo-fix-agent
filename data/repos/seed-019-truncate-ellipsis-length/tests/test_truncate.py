import pytest

from textutil.cards import card_title
from textutil.truncate import truncate


def test_short_text_unchanged():
    assert truncate("hello", 10) == "hello"
    assert truncate("exactly10!", 10) == "exactly10!"


def test_rejects_tiny_limit():
    with pytest.raises(ValueError):
        truncate("abc", 2, ellipsis="...")


def test_card_title_collapses_whitespace():
    assert card_title("  Build   failed \n on main ") == "Build failed on main"
