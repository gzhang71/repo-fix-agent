from datetime import datetime, timezone


def parse_event(line):
    """'2024-05-01T23:30:00Z,signup' -> (aware UTC datetime, kind)."""
    ts, kind = line.strip().split(",", 1)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc), kind
