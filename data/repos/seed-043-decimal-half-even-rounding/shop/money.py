from decimal import Decimal

CENT = Decimal("0.01")


def to_money(value):
    """Coerce int/str/Decimal to a Decimal rounded to cents (half-up, like the POS)."""
    return Decimal(str(value)).quantize(CENT)
