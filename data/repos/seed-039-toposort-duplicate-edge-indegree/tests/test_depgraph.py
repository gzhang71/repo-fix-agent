import pytest

from depgraph.manifest import build_graph, install_order
from depgraph.toposort import CycleError


def test_simple_chain():
    assert install_order({"app": ["lib"], "lib": ["core"], "core": []}) == ["core", "lib", "app"]


def test_diamond():
    m = {"app": ["a", "b"], "a": ["base"], "b": ["base"], "base": []}
    assert install_order(m) == ["base", "a", "b", "app"]


def test_real_cycle():
    with pytest.raises(CycleError):
        install_order({"a": ["b"], "b": ["a"]})


def test_dependents():
    g = build_graph({"app": ["lib"], "cli": ["lib"]})
    assert g.dependents("lib") == ["app", "cli"]
