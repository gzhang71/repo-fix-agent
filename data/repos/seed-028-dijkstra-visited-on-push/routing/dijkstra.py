import heapq


def shortest_paths(graph, source):
    """Return (dist, prev) maps for every node reachable from source."""
    dist = {source: 0}
    prev = {source: None}
    visited = {source}
    heap = [(0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        for nbr, w in graph.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            nd = d + w
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(heap, (nd, nbr))
    return dist, prev


def route(graph, source, target):
    """Return (path, cost) of the cheapest route, or (None, inf) if unreachable."""
    dist, prev = shortest_paths(graph, source)
    if target not in dist:
        return None, float("inf")
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1], dist[target]
