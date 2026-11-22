from decimal import Decimal

import pytest

from fx.convert import convert


def test_same_currency_is_identity():
    assert convert("12.34", "EUR", "EUR") == Decimal("12.34")


def test_unknown_currency():
    with pytest.raises(ValueError):
        convert(1, "USD", "XYZ")


def test_round_trip():
    there = convert(100, "EUR", "GBP")
    assert convert(there, "GBP", "EUR") == Decimal("100.00")
