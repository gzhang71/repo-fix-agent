class IndexedHeap:
    """Binary min-heap of (priority, key) with O(log n) priority updates by key."""

    def __init__(self):
        self._heap = []  # list of [priority, key]
        self._pos = {}   # key -> index in self._heap

    def __len__(self):
        return len(self._heap)

    def __contains__(self, key):
        return key in self._pos

    def push(self, key, priority):
        if key in self._pos:
            raise KeyError(f"{key!r} is already queued")
        self._heap.append([priority, key])
        self._pos[key] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def pop(self):
        """Remove and return (key, priority) with the smallest priority."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        top = self._heap[0]
        last = self._heap.pop()
        del self._pos[top[1]]
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return top[1], top[0]

    def update(self, key, priority):
        i = self._pos[key]
        old = self._heap[i][0]
        self._heap[i][0] = priority
        if priority < old:
            self._sift_up(i)
        else:
            self._sift_down(i)

    def _swap(self, i, j):
        h = self._heap
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][1]] = i
        self._pos[h[j][1]] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i][0] < self._heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            smallest = i
            for child in (2 * i + 1, 2 * i + 2):
                if child < n and self._heap[child][0] < self._heap[smallest][0]:
                    smallest = child
            if smallest == i:
                return
            self._swap(i, smallest)
            i = smallest
