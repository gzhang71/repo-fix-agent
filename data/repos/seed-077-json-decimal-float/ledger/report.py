from decimal import ROUND_HALF_UP, Decimal

from .codec import load_entries


def balance(text, opening=Decimal("0.00")):
    total = opening
    for entry in load_entries(text):
        total += entry["amount"]
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
