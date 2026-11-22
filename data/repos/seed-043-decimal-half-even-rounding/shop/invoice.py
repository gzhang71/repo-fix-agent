from decimal import Decimal

from .money import to_money


def line_total(unit_price, qty, discount_pct=0):
    gross = Decimal(str(unit_price)) * qty
    discount = gross * Decimal(str(discount_pct)) / 100
    return to_money(gross - discount)


def invoice_total(lines):
    return sum((line_total(*line) for line in lines), Decimal("0.00"))
