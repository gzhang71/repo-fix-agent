def to_bool(raw):
    """Parse an environment-variable string as a boolean."""
    return bool(raw.strip())


def to_int(raw):
    return int(raw.strip())
