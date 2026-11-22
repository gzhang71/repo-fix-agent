from .coerce import to_bool, to_int

DEFAULTS = {"debug": False, "workers": 4, "cache_enabled": True, "region": "local"}
PREFIX = "APP_"


def load_settings(environ):
    """Build settings from DEFAULTS, overridden by APP_* variables in `environ`."""
    settings = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        raw = environ.get(PREFIX + key.upper())
        if raw is None:
            continue
        if isinstance(default, bool):
            settings[key] = to_bool(raw)
        elif isinstance(default, int):
            settings[key] = to_int(raw)
        else:
            settings[key] = raw
    return settings
