from .histogram import histogram


def ascii_hist(values, bins, width=20):
    counts = histogram(values, bins)
    peak = max(counts) or 1
    return "\n".join("#" * round(c * width / peak) for c in counts)
