from .escapes import unescape


def _logical_lines(text):
    """Join physical lines ending in a backslash with the following line."""
    buf = None
    for raw in text.splitlines():
        line = raw.lstrip() if buf is not None else raw
        if line.endswith("\\"):
            buf = (buf or "") + line[:-1]
            continue
        yield (buf or "") + line
        buf = None
    if buf is not None:
        yield buf


def _split_key_value(line):
    """Split at the first unescaped '=' or ':'."""
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "=:":
            return line[:i].rstrip(), line[i + 1:].lstrip()
        i += 1
    return line, ""


def load(text):
    props = {}
    for line in _logical_lines(text):
        stripped = line.strip()
        if not stripped or stripped[0] in "#!":
            continue
        key, value = _split_key_value(stripped)
        props[unescape(key)] = unescape(value)
    return props
