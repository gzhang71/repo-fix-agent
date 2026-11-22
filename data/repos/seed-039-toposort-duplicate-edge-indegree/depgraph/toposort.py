from collections import deque


class CycleError(Exception):
    pass


def topo_order(graph):
    """Kahn's algorithm; ties broken alphabetically for a deterministic order."""
    indeg = dict(graph.indegree)
    ready = deque(sorted(n for n in graph.nodes if indeg[n] == 0))
    order = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for succ in sorted(graph.successors.get(node, ())):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                ready.append(succ)
    if len(order) != len(graph.nodes):
        stuck = sorted(n for n in graph.nodes if indeg[n] > 0)
        raise CycleError(f"dependency cycle among: {', '.join(stuck)}")
    return order
