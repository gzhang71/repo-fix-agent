from .money import from_cents, to_cents


def allocate(total, ratios):
    """Split `total` across `ratios` proportionally, to the cent.

    The parts always sum exactly to `total`.
    """
    if not ratios or any(r < 0 for r in ratios) or sum(ratios) == 0:
        raise ValueError("ratios must be non-negative and not all zero")
    cents = to_cents(total)
    weight = sum(ratios)
    parts = [cents * r // weight for r in ratios]
    return [from_cents(p) for p in parts]


def split_evenly(total, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    return allocate(total, [1] * n)
