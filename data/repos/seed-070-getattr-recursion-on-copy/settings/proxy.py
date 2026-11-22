class SettingsProxy:
    """Read-only view over a settings dict; attribute access maps to keys.

    Keys missing from `data` fall back to `defaults`.
    """

    def __init__(self, data, defaults=None):
        self._data = dict(data)
        self._defaults = dict(defaults or {})

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails.
        if name in self._data:
            return self._data[name]
        if name in self._defaults:
            return self._defaults[name]
        raise AttributeError(f"no setting named {name!r}")

    def with_overrides(self, **overrides):
        return SettingsProxy({**self._data, **overrides}, self._defaults)
