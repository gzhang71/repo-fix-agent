from datetime import datetime, timedelta

from jobs.durations import format_duration
from jobs.summary import describe_run


def test_hours_and_minutes():
    assert format_duration(timedelta(hours=2, minutes=5)) == "2h 5m"


def test_zero_and_seconds():
    assert format_duration(timedelta(0)) == "0s"
    assert format_duration(timedelta(seconds=45)) == "45s"


def test_describe_run():
    assert describe_run("etl", datetime(2024, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 30)) == "etl took 30m"
