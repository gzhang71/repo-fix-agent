from datetime import date

from analytics.report import weekly_counts
from analytics.weeks import week_key, week_start


def test_week_key_mid_year():
    assert week_key(date(2024, 2, 14)) == "2024-W07"


def test_week_start_is_monday():
    assert week_start(date(2024, 2, 15)) == date(2024, 2, 12)


def test_weekly_counts():
    assert weekly_counts([date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 11)]) == {
        "2024-W10": 2, "2024-W11": 1}
