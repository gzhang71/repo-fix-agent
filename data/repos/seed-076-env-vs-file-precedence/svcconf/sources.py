PREFIX = "SVC_"


def from_env(environ):
    """SVC_FOO=bar -> {"foo": "bar"}."""
    return {k[len(PREFIX):].lower(): v for k, v in environ.items() if k.startswith(PREFIX)}


def from_file_text(text):
    """Parse `key = value` lines; '#' starts a comment."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out
