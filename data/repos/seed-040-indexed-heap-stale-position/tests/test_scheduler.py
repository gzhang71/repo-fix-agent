import pytest

from taskq.heap import IndexedHeap
from taskq.scheduler import Scheduler


def test_runs_in_deadline_order():
    s = Scheduler()
    for task, deadline in [("c", 30), ("a", 10), ("d", 40), ("b", 20)]:
        s.submit(task, deadline)
    assert s.run_all() == ["a", "b", "c", "d"]


def test_reschedule_before_any_pop():
    s = Scheduler()
    s.submit("a", 10)
    s.submit("b", 20)
    s.submit("c", 30)
    s.reschedule("c", 5)
    s.reschedule("a", 25)
    assert s.run_all() == ["c", "b", "a"]


def test_duplicate_push_rejected():
    h = IndexedHeap()
    h.push("x", 1)
    with pytest.raises(KeyError):
        h.push("x", 2)


def test_pop_empty():
    with pytest.raises(IndexError):
        IndexedHeap().pop()
