import pytest

from maze.grid import parse
from maze.solver import path_length, shortest_path

WALLED = """
#####
#S..#
#.#.#
#..E#
#####
"""


def test_walled_maze():
    assert path_length(WALLED) == 4
    assert shortest_path(WALLED)[0] == (1, 1)


def test_unreachable():
    assert shortest_path("#####\n#S#E#\n#####") is None


def test_ragged_rows_rejected():
    with pytest.raises(ValueError):
        parse("..\n...")
