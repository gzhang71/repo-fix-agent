import pytest

from pkgmeta.release import next_version
from pkgmeta.version import InvalidVersion, is_valid_version, parse_version


def test_parse():
    assert parse_version(" 1.10.3\n") == (1, 10, 3)


def test_invalid():
    assert not is_valid_version("1.2")
    with pytest.raises(InvalidVersion):
        parse_version("v1.2.3")


def test_next_version():
    assert next_version("1.4.2") == "1.4.3"
    assert next_version("1.4.2", "minor") == "1.5.0"
