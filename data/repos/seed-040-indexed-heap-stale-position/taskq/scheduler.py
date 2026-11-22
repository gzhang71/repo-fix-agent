from .heap import IndexedHeap


class Scheduler:
    """Runs tasks in deadline order; deadlines can be moved while a task is queued."""

    def __init__(self):
        self._queue = IndexedHeap()

    def submit(self, task, deadline):
        self._queue.push(task, deadline)

    def reschedule(self, task, deadline):
        self._queue.update(task, deadline)

    def next_task(self):
        return self._queue.pop()[0]

    def pending(self):
        return len(self._queue)

    def run_all(self):
        out = []
        while self.pending():
            out.append(self.next_task())
        return out
