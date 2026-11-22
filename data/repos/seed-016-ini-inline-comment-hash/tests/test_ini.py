import pytest

from iniparse import ParseError, parse_ini
from iniparse.settings import Settings


def test_sections_and_keys():
    data = parse_ini("[general]\nName = demo\n[db]\nport=5432\n")
    assert data == {"general": {"name": "demo"}, "db": {"port": "5432"}}


def test_comment_lines_ignored():
    assert parse_ini("# top\n[a]\n; note\nx = 1\n") == {"a": {"x": "1"}}


def test_inline_comment_after_value():
    assert parse_ini("[a]\nx = 1  # one\n") == {"a": {"x": "1"}}


def test_errors():
    with pytest.raises(ParseError, match="line 1"):
        parse_ini("x = 1\n")
    with pytest.raises(ParseError, match="line 2"):
        parse_ini("[a]\njunk\n")


def test_settings_get_default():
    assert Settings("[general]\n").get("missing", default=3) == 3
