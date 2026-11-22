from .extract import find_amounts


def line_amount(line):
    """The price of a receipt line is the last amount on it (quantities come first)."""
    amounts = find_amounts(line)
    return amounts[-1] if amounts else None


def receipt_total(lines):
    total = 0.0
    for line in lines:
        amount = line_amount(line)
        if amount is not None:
            total += amount
    return round(total, 2)
