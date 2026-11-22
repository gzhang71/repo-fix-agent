from .distance import levenshtein


def closest_with_distance(word, candidates, max_distance=2):
    best = None
    for c in candidates:
        d = levenshtein(word.lower(), c.lower())
        if d <= max_distance and (best is None or d < best[0]):
            best = (d, c)
    return best


def closest(word, candidates, max_distance=2):
    """Best candidate within max_distance edits, or None."""
    best = closest_with_distance(word, candidates, max_distance)
    return best[1] if best else None
