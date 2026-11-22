import pytest

from metrics.percentile import percentile
from metrics.summary import latency_summary


def test_minimum():
    assert percentile([3, 1, 2], 0) == 1
    assert percentile([5], 0) == 5


def test_validates_p():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summary_shape():
    s = latency_summary(list(range(40, 0, -1)))
    assert s["count"] == 40 and s["max"] == 40
