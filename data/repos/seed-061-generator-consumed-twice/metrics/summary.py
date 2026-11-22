def summarize(samples):
    """samples: iterable of (endpoint, ms). Returns count, mean, max and the slowest endpoint."""
    count = sum(1 for _ in samples)
    if count == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "slowest": None}
    total = sum(ms for _, ms in samples)
    slowest = max(samples, key=lambda s: s[1])
    return {"count": count, "mean": total / count, "max": slowest[1], "slowest": slowest[0]}
