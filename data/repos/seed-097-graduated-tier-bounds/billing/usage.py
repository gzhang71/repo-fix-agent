def total_units(events, customer, period):
    """events: iterable of (customer, period, units) tuples."""
    return sum(units for c, p, units in events if c == customer and p == period)
