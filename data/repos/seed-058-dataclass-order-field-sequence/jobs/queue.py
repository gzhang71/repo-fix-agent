import heapq
import itertools

from .model import Job


class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def submit(self, name, priority=5):
        job = Job(name, priority, next(self._counter))
        heapq.heappush(self._heap, job)
        return job

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty JobQueue")
        return heapq.heappop(self._heap)

    def drain(self):
        out = []
        while self._heap:
            out.append(self.pop().name)
        return out

    def __len__(self):
        return len(self._heap)
