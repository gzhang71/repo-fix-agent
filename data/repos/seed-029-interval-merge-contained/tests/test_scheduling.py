from scheduling.availability import free_slots
from scheduling.intervals import merge


def test_merge_overlapping():
    assert merge([(5, 8), (1, 3), (2, 4)]) == [(1, 4), (5, 8)]


def test_merge_touching():
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]


def test_free_slots_basic():
    busy = [(600, 660), (720, 780)]
    assert free_slots(busy, 540, 1020) == [(540, 600), (660, 720), (780, 1020)]


def test_free_slots_min_length():
    busy = [(600, 610)]
    assert free_slots(busy, 540, 1020, min_length=480) == []
