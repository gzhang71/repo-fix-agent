from collections import Counter

from .weeks import week_key


def weekly_counts(dates):
    return dict(sorted(Counter(week_key(d) for d in dates).items()))
