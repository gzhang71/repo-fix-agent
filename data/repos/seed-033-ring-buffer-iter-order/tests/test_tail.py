import pytest

from logtail.ringbuffer import RingBuffer
from logtail.tail import tail, tail_matching


def test_short_log():
    assert tail(["a\n", "b\n"], n=5) == ["a", "b"]


def test_latest_after_wrap():
    buf = RingBuffer(3)
    for i in range(7):
        buf.append(i)
    assert buf.latest() == 6
    assert len(buf) == 3


def test_tail_matching_short():
    assert tail_matching(["ok", "ERR 1", "ok", "ERR 2"], "ERR", n=5) == ["ERR 1", "ERR 2"]


def test_bad_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)
