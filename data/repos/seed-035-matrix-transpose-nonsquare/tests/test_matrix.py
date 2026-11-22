import pytest

from matrix.core import identity, shape, transpose
from matrix.linalg import matmul, trace


def test_transpose_square():
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_matmul_identity():
    m = [[1, 2], [3, 4]]
    assert matmul(m, identity(2)) == m


def test_shape_mismatch():
    with pytest.raises(ValueError):
        matmul([[1, 2]], [[1, 2]])


def test_trace_and_shape():
    assert trace(identity(3)) == 3
    assert shape([[1, 2, 3]]) == (1, 3)
