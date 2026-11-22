from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def q(x):
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def add_tax(net, rate):
    """Gross price for a net price and a rate like Decimal('0.20')."""
    return q(Decimal(str(net)) * (1 + Decimal(str(rate))))


def net_from_gross(gross, rate):
    """Net price contained in a tax-inclusive gross price."""
    return q(gross * (1 - rate))
