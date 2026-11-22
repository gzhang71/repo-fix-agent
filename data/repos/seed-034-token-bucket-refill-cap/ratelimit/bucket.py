class TokenBucket:
    def __init__(self, capacity, refill_rate, clock):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self._tokens = self._tokens + elapsed * self.refill_rate

    def try_acquire(self, n=1):
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    @property
    def available(self):
        self._refill()
        return self._tokens
