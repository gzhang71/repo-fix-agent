def merge(intervals):
    """Merge overlapping or touching (start, end) intervals. Returns a sorted list."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def total_length(intervals):
    """Total time covered by the intervals, counting overlaps once."""
    return sum(end - start for start, end in merge(intervals))
