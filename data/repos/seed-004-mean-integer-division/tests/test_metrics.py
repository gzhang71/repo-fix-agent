import pytest
from stats.metrics import mean, median


def test_mean_of_evenly_divisible():
    assert mean([2, 4, 6]) == 4


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])
