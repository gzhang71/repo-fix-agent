def allocate(total_cents, weights):
    """Split `total_cents` across lines in proportion to `weights`.

    The parts always add up to exactly `total_cents`.
    """
    weight_sum = sum(weights)
    if weight_sum == 0:
        return [0] * len(weights)
    return [round(total_cents * w / weight_sum) for w in weights]
