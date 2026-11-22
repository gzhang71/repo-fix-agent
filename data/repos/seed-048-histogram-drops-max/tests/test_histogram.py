import pytest

from charts.histogram import bin_edges, histogram


def test_interior_values():
    assert histogram([0.5, 1.5, 2.5], 3, 0, 3) == [1, 1, 1]


def test_out_of_range_ignored():
    assert histogram([-1, 0.5, 5], 2, 0, 2) == [1, 0]


def test_bin_edges():
    assert bin_edges(0, 10, 2) == [0, 5, 10]


def test_invalid_bounds():
    with pytest.raises(ValueError):
        histogram([1, 1], 2)
