from .bucket import TokenBucket


class RateLimiter:
    """One token bucket per client key."""

    def __init__(self, capacity, refill_rate, clock):
        self._args = (capacity, refill_rate, clock)
        self._buckets = {}

    def allow(self, key):
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(*self._args)
        return self._buckets[key].try_acquire()
