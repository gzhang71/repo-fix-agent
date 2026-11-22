import math


class RunningStats:
    """Streaming mean/variance (Welford). Shards are combined with merge()."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def merge(self, other):
        """Return new stats equivalent to having pushed both streams."""
        out = RunningStats()
        out.n = self.n + other.n
        if out.n == 0:
            return out
        delta = other.mean - self.mean
        out.mean = self.mean + delta * other.n / out.n
        out.m2 = self.m2 + other.m2
        return out

    @property
    def variance(self):
        """Sample variance (n - 1 denominator)."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def stddev(self):
        return math.sqrt(self.variance)
