from collections import ChainMap

from .sources import from_env, from_file_text

DEFAULTS = {"log_level": "warning", "port": "8080", "region": "local"}


def build_settings(file_text="", environ=None, overrides=None):
    """Resolve settings. Precedence, highest first:
    explicit overrides > environment (SVC_*) > config file > DEFAULTS.
    """
    chain = ChainMap(overrides or {}, from_file_text(file_text),
                     from_env(environ or {}), DEFAULTS)
    return dict(chain)
