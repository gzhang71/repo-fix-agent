from netgraph.clusters import clusters, largest_cluster_size
from netgraph.dsu import DisjointSet


def test_clusters():
    hosts = ["a", "b", "c", "d", "e"]
    links = [("a", "b"), ("d", "c"), ("b", "c")]
    assert clusters(hosts, links) == [["a", "b", "c", "d"], ["e"]]


def test_connected():
    ds = DisjointSet()
    ds.union(1, 2)
    ds.union(3, 4)
    assert ds.connected(2, 1)
    assert not ds.connected(1, 3)


def test_largest_cluster_empty():
    assert largest_cluster_size([], []) == 0
