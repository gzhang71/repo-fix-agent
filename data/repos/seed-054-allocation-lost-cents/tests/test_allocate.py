from decimal import Decimal

import pytest

from payments.allocate import allocate, split_evenly
from payments.installments import schedule
from payments.money import from_cents, to_cents


def test_cents_round_trip():
    assert to_cents("12.345") == 1235
    assert from_cents(1235) == Decimal("12.35")


def test_even_split_divisible():
    assert split_evenly(Decimal("100.00"), 2) == [Decimal("50.00"), Decimal("50.00")]


def test_weighted_divisible():
    assert allocate(Decimal("10.00"), [3, 1]) == [Decimal("7.50"), Decimal("2.50")]


def test_rejects_bad_ratios():
    with pytest.raises(ValueError):
        allocate(Decimal("1.00"), [0, 0])


def test_schedule_numbering():
    assert [m for m, _ in schedule(Decimal("90.00"), 3)] == [1, 2, 3]
