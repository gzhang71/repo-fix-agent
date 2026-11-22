from .dates import add_months


def renewal_dates(start, count, interval_months=1):
    """Next `count` renewal dates, always computed from the anchor to avoid drift."""
    return [add_months(start, interval_months * i) for i in range(1, count + 1)]
