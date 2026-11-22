from collections import defaultdict

from .ranges import date_range


def daily_totals(records, start, end):
    """records: iterable of (date, amount). Returns {date: total} for every day in range."""
    totals = defaultdict(float)
    for day, amount in records:
        totals[day] += amount
    return {day: totals.get(day, 0.0) for day in date_range(start, end)}
