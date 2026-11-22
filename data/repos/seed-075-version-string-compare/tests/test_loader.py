from pluginhost.loader import load_plugins
from pluginhost.versions import satisfies


def test_satisfies_basic():
    assert satisfies("1.2.0", "1.1.0")
    assert satisfies("v2.0.0", "2.0.0")
    assert not satisfies("1.0.0", "1.2.0")


def test_load_plugins():
    loaded, skipped = load_plugins(
        [{"name": "a", "requires_host": "1.1.0"}, {"name": "b", "requires_host": "2.0.0"}, {"name": "c"}],
        "1.5.0")
    assert loaded == ["a", "c"]
    assert skipped == {"b": "requires host >= 2.0.0"}
