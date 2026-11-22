import pytest

from minitpl import render
from minitpl.escape import escape_html


def test_plain_substitution():
    assert render("Hi {{ name }}!", {"name": "Ann"}) == "Hi Ann!"


def test_ampersand():
    assert escape_html("Tom & Jerry") == "Tom &amp; Jerry"


def test_safe_filter_is_raw():
    assert render("{{ html|safe }}", {"html": "<b>x</b>"}) == "<b>x</b>"


def test_missing_variable():
    with pytest.raises(KeyError):
        render("{{ nope }}", {})
