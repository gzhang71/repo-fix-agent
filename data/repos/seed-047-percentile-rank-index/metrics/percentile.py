import math


def percentile(values, p):
    """p-th percentile (0-100), linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of empty data")
    if not 0 <= p <= 100:
        raise ValueError("p must be within [0, 100]")
    s = sorted(values)
    k = len(s) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
