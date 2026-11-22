from datetime import date

from billing.dates import add_months, month_start
from billing.subscriptions import renewal_dates


def test_add_months_simple():
    assert add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)


def test_add_months_across_year():
    assert add_months(date(2023, 11, 10), 3) == date(2024, 2, 10)
    assert add_months(date(2024, 3, 10), -3) == date(2023, 12, 10)


def test_month_start():
    assert month_start(date(2024, 7, 19)) == date(2024, 7, 1)


def test_renewal_dates_mid_month():
    assert renewal_dates(date(2024, 1, 10), 3) == [
        date(2024, 2, 10), date(2024, 3, 10), date(2024, 4, 10)]
