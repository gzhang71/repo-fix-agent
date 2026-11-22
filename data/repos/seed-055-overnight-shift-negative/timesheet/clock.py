def parse_hhmm(text):
    """'HH:MM' -> minutes since midnight."""
    h, m = (int(part) for part in text.strip().split(":"))
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"invalid time {text!r}")
    return h * 60 + m


def shift_minutes(start, end):
    """Length of a shift from 'HH:MM' clock-in/out times. Shifts may cross midnight."""
    return parse_hhmm(end) - parse_hhmm(start)
