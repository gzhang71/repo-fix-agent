from collections import Counter

from .parse import parse_event
from .tz import local_date


def daily_counts(lines, offset_minutes=0, kind=None):
    counts = Counter()
    for line in lines:
        dt, k = parse_event(line)
        if kind is None or k == kind:
            counts[local_date(dt, offset_minutes).isoformat()] += 1
    return dict(sorted(counts.items()))
