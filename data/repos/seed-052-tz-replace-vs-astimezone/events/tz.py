from datetime import timedelta, timezone


def fixed_offset(minutes):
    """tzinfo for a fixed UTC offset in minutes, e.g. 330 (IST) or -300 (EST)."""
    return timezone(timedelta(minutes=minutes))


def to_local(dt, offset_minutes):
    """Express an aware datetime in the customer's local (fixed-offset) time."""
    return dt.replace(tzinfo=fixed_offset(offset_minutes))


def local_date(dt, offset_minutes):
    return to_local(dt, offset_minutes).date()
