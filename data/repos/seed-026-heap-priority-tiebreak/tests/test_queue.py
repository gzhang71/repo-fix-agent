import pytest

from jobs import JobQueue
from jobs.models import Job
from jobs.worker import run_all


def test_lower_priority_first():
    q = JobQueue()
    q.push(Job("c"), priority=3)
    q.push(Job("a"), priority=1)
    q.push(Job("b"), priority=2)
    assert run_all(q, lambda j: j.name) == ["a", "b", "c"]
    assert len(q) == 0


def test_pop_empty_raises():
    with pytest.raises(IndexError):
        JobQueue().pop()
