import pytest

from jobs.queue import JobQueue


def test_submit_assigns_sequence():
    q = JobQueue()
    a = q.submit("a")
    b = q.submit("b")
    assert (a.seq, b.seq) == (0, 1)
    assert len(q) == 2


def test_pop_empty_raises():
    with pytest.raises(IndexError):
        JobQueue().pop()


def test_drain_empties_queue():
    q = JobQueue()
    q.submit("only", priority=3)
    assert q.drain() == ["only"]
    assert len(q) == 0
