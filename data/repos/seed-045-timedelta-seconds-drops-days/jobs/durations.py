def format_duration(td):
    """Human-readable duration such as '1d 2h 5m' or '45s'."""
    secs = td.seconds
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    units = ((days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s"))
    parts = [f"{value}{unit}" for value, unit in units if value]
    return " ".join(parts) or "0s"
