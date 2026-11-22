# Rates in basis points (1/100 of a percent).
RATES_BPS = {"CA": 725, "NY": 400, "OR": 0}


def tax_rate_bps(region):
    try:
        return RATES_BPS[region]
    except KeyError:
        raise ValueError(f"no tax rate for region {region!r}") from None


def compute_tax(order):
    """Sales tax on goods: charged on the discounted price, never on shipping."""
    base = order.subtotal_cents
    return (base * tax_rate_bps(order.region) + 5000) // 10000
