from app.config import DEFAULTS, load_config


def test_no_overrides_returns_defaults():
    assert load_config() == DEFAULTS


def test_defaults_not_mutated():
    load_config()["timeout"] = 1
    assert DEFAULTS["timeout"] == 30
