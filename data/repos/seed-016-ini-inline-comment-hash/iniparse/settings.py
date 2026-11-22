import re

from .parser import parse_ini

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


class Settings:
    def __init__(self, text):
        self._data = parse_ini(text)

    def get(self, key, section="general", default=None):
        return self._data.get(section, {}).get(key, default)

    def color(self, key):
        value = self.get(key, section="theme", default="")
        if not _HEX.fullmatch(value):
            raise ValueError(f"invalid colour {value!r}")
        return value
