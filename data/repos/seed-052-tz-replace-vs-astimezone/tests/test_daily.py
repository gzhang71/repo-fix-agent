from datetime import datetime, timedelta, timezone

from events.daily import daily_counts
from events.parse import parse_event
from events.tz import fixed_offset


def test_parse_converts_to_utc():
    dt, kind = parse_event("2024-05-01T23:30:00+02:00,signup")
    assert dt == datetime(2024, 5, 1, 21, 30, tzinfo=timezone.utc)
    assert kind == "signup"


def test_fixed_offset():
    assert fixed_offset(330).utcoffset(None) == timedelta(hours=5, minutes=30)


def test_daily_counts_utc():
    lines = ["2024-05-01T10:00:00Z,signup", "2024-05-01T23:59:00Z,signup",
             "2024-05-02T00:01:00Z,login"]
    assert daily_counts(lines) == {"2024-05-01": 2, "2024-05-02": 1}
    assert daily_counts(lines, kind="login") == {"2024-05-02": 1}
