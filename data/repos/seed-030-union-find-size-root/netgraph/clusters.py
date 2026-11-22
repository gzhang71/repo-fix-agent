from .dsu import DisjointSet


def _build(hosts, links):
    ds = DisjointSet()
    for h in hosts:
        ds.add(h)
    for a, b in links:
        ds.union(a, b)
    return ds


def clusters(hosts, links):
    """Connected groups of hosts, largest first."""
    ds = _build(hosts, links)
    groups = {}
    for h in hosts:
        groups.setdefault(ds.find(h), []).append(h)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def largest_cluster_size(hosts, links):
    ds = _build(hosts, links)
    return max((ds.component_size(h) for h in hosts), default=0)
