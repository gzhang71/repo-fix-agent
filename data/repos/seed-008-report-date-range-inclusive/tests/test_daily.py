from datetime import date

from reports.daily import daily_totals


def test_sums_per_day():
    recs = [(date(2024, 3, 1), 5.0), (date(2024, 3, 1), 2.5)]
    out = daily_totals(recs, date(2024, 3, 1), date(2024, 3, 2))
    assert out[date(2024, 3, 1)] == 7.5
