import pytest

from settings.proxy import SettingsProxy
from settings.runtime import BASE


def test_attribute_access_and_defaults():
    assert BASE.workers == 4
    assert BASE.timeout == 30


def test_unknown_setting():
    with pytest.raises(AttributeError, match="nope"):
        BASE.nope


def test_with_overrides():
    s = SettingsProxy({"a": 1}, {"b": 2}).with_overrides(a=5)
    assert (s.a, s.b) == (5, 2)
