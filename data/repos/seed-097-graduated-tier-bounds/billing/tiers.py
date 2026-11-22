def graduated_cost(units, tiers):
    """Price each unit at the rate of the tier it falls in (like tax brackets).

    Tier bounds are cumulative: after Tier(1000, 0), Tier(10000, 2) covers units
    1,001-10,000.
    """
    total = 0
    remaining = units
    for tier in tiers:
        if remaining <= 0:
            break
        in_tier = remaining if tier.up_to is None else min(remaining, tier.up_to)
        total += in_tier * tier.unit_cents
        remaining -= in_tier
    return total


def volume_cost(units, tiers):
    """All units are priced at the rate of the single tier the total falls in."""
    for tier in tiers:
        if tier.up_to is None or units <= tier.up_to:
            return units * tier.unit_cents
    raise ValueError("tiers must end with an unbounded tier")


def usage_cost(units, plan):
    if plan.mode == "graduated":
        return graduated_cost(units, plan.tiers)
    if plan.mode == "volume":
        return volume_cost(units, plan.tiers)
    raise ValueError(f"unknown pricing mode {plan.mode!r}")
