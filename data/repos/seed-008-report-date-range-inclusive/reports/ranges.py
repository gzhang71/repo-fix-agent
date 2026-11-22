from datetime import timedelta


def date_range(start, end):
    """Yield each date from start to end, inclusive."""
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)
