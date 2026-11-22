from alerts.scheduler import Reminder
from alerts.timeparse import parse_ts


def test_parse_naive_keeps_wall_clock():
    dt = parse_ts("2024-03-10T09:00:00")
    assert (dt.hour, dt.minute) == (9, 0)


def test_lead_time():
    r = Reminder("x", "2024-03-10T09:00:00", lead_minutes=15)
    assert r.fire_time().strftime("%H:%M") == "08:45"
