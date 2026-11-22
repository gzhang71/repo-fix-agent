class RingBuffer:
    """Fixed-capacity buffer that keeps the most recent `capacity` items."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._items = [None] * capacity
        self._start = 0
        self._size = 0

    def append(self, item):
        end = (self._start + self._size) % self.capacity
        self._items[end] = item
        if self._size < self.capacity:
            self._size += 1
        else:
            self._start = (self._start + 1) % self.capacity

    def latest(self):
        if not self._size:
            raise IndexError("empty buffer")
        return self._items[(self._start + self._size - 1) % self.capacity]

    def __len__(self):
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield self._items[i]
