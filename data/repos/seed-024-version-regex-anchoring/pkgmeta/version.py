import re

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class InvalidVersion(ValueError):
    pass


def parse_version(text):
    """Parse 'MAJOR.MINOR.PATCH' into a tuple of ints."""
    m = _VERSION.match(text.strip())
    if not m:
        raise InvalidVersion(text)
    return tuple(int(g) for g in m.groups())


def is_valid_version(text):
    try:
        parse_version(text)
    except InvalidVersion:
        return False
    return True
