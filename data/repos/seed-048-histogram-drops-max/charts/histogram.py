def bin_edges(lo, hi, bins):
    width = (hi - lo) / bins
    return [lo + i * width for i in range(bins + 1)]


def histogram(values, bins, lo=None, hi=None):
    """Counts per equal-width bin over [lo, hi]; the last bin includes hi."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    values = list(values)
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        if 0 <= idx < bins:
            counts[idx] += 1
    return counts
