from decimal import Decimal

from shop.invoice import invoice_total, line_total
from shop.money import to_money


def test_to_money_rounds_to_cents():
    assert to_money("19.999") == Decimal("20.00")
    assert to_money(3) == Decimal("3.00")


def test_line_total():
    assert line_total("10.00", 3) == Decimal("30.00")
    assert line_total("10.00", 2, 25) == Decimal("15.00")


def test_invoice_total():
    assert invoice_total([("1.10", 2), ("0.30", 1)]) == Decimal("2.50")
