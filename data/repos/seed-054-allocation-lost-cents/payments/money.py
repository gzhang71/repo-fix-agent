from decimal import ROUND_HALF_UP, Decimal


def to_cents(amount):
    return int((Decimal(str(amount)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def from_cents(cents):
    return (Decimal(cents) / 100).quantize(Decimal("0.01"))
