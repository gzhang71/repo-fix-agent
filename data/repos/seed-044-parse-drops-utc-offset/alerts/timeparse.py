from datetime import datetime, timezone


def parse_ts(text):
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Strings without an offset are assumed to already be in UTC.
    """
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt
