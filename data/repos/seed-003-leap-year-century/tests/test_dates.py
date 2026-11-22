from calendar_utils.dates import days_in_month, is_leap_year


def test_common_years():
    assert days_in_month(2023, 2) == 28
    assert days_in_month(1900, 2) == 28


def test_regular_leap_year():
    assert is_leap_year(2024)
    assert days_in_month(2024, 4) == 30
