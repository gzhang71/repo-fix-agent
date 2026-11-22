import pytest

from routing.dijkstra import route
from routing.graph import Graph


def test_line_graph():
    g = Graph()
    g.add_edge("A", "B", 2)
    g.add_edge("B", "C", 3)
    assert route(g, "A", "C") == (["A", "B", "C"], 5)


def test_unreachable():
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("X", "Y", 1)
    assert route(g, "A", "Y") == (None, float("inf"))


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        Graph().add_edge("A", "B", -1)
