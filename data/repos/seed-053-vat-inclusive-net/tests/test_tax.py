from decimal import Decimal

from pos.receipt import receipt_line, vat_breakdown
from pos.tax import add_tax


def test_add_tax():
    assert add_tax("100.00", "0.20") == Decimal("120.00")
    assert add_tax("9.99", "0.07") == Decimal("10.69")


def test_zero_rated():
    assert vat_breakdown("12.50", "0") == {
        "gross": Decimal("12.50"), "net": Decimal("12.50"), "tax": Decimal("0.00")}


def test_receipt_line_format():
    assert receipt_line("Book", "7.00", "0").startswith("Book: net 7.00")
