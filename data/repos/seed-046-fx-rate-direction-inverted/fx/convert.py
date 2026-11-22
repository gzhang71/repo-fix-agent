from decimal import ROUND_HALF_UP, Decimal

from .rates import rate


def convert(amount, src, dst):
    amount = Decimal(str(amount))
    if src == dst:
        return amount
    usd = amount / rate(src)
    return (usd * rate(dst)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
