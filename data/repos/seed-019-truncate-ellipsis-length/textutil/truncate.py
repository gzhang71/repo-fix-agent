ELLIPSIS = "\u2026"


def truncate(text, max_len, ellipsis=ELLIPSIS):
    """Shorten text to at most `max_len` characters, marking the cut with `ellipsis`."""
    if max_len < len(ellipsis):
        raise ValueError("max_len is shorter than the ellipsis")
    if len(text) <= max_len:
        return text
    return text[:max_len] + ellipsis
