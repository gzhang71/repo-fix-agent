import pytest

from propfile import Config, load


def test_basic_and_comments():
    text = "# comment\n! also comment\nname = demo\nport: 8080\n\n"
    assert load(text) == {"name": "demo", "port": "8080"}


def test_continuation_lines():
    text = "colors = red, \\\n    green, \\\n    blue\nnext = 1\n"
    cfg = Config(text)
    assert cfg.get_list("colors") == ["red", "green", "blue"]
    assert cfg.get("next") == "1"


def test_escapes():
    assert load(r"msg = a\tb\=c" + "\n" + r"k\:x = 1") == {"msg": "a\tb=c", "k:x": "1"}


def test_require_missing():
    with pytest.raises(KeyError, match="output.name"):
        Config("a = 1\n").require("output.name")
