_REPLACEMENTS = [
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
    ("'", "&#x27;"),
    ("&", "&amp;"),
]


def escape_html(value):
    """Escape a value for safe inclusion in HTML text or attribute values."""
    s = str(value)
    for old, new in _REPLACEMENTS:
        s = s.replace(old, new)
    return s
