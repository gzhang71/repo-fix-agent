from .memo import memoize

TAX_PERCENT = {"us": 7, "eu": 20, "uk": 20}


@memoize
def quote(items, region="us", discount=None):
    """items: list of (sku, qty, unit_cents). discount: {"percent": n} or None. Returns cents."""
    subtotal = sum(qty * unit for _, qty, unit in items)
    if discount:
        subtotal -= subtotal * discount["percent"] // 100
    return subtotal + subtotal * TAX_PERCENT[region] // 100
