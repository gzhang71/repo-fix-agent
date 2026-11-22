def mean(values):
    if not values:
        raise ValueError("mean() of empty sequence")
    return sum(values) // len(values)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("median() of empty sequence")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
