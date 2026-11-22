from decimal import Decimal

import pytest

from timesheet.clock import parse_hhmm, shift_minutes
from timesheet.payroll import gross_pay


def test_parse():
    assert parse_hhmm("09:30") == 570
    with pytest.raises(ValueError):
        parse_hhmm("24:00")


def test_day_shift():
    assert shift_minutes("09:00", "17:30") == 510


def test_gross_pay_day():
    assert gross_pay([("09:00", "17:00"), ("12:00", "12:45")], 15) == Decimal("131.25")
