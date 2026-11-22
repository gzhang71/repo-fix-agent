import heapq


class JobQueue:
    """Min-priority queue of jobs: a lower priority number runs first."""

    def __init__(self):
        self._heap = []

    def push(self, job, priority=0):
        heapq.heappush(self._heap, (priority, job))

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty JobQueue")
        return heapq.heappop(self._heap)[1]

    def __len__(self):
        return len(self._heap)

    def drain(self):
        while self._heap:
            yield self.pop()
