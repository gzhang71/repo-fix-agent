from collections import defaultdict


class Graph:
    def __init__(self):
        self._adj = defaultdict(list)

    def add_edge(self, u, v, weight, bidirectional=True):
        if weight < 0:
            raise ValueError("negative edge weights are not supported")
        self._adj[u].append((v, weight))
        if bidirectional:
            self._adj[v].append((u, weight))

    def neighbors(self, node):
        return list(self._adj.get(node, ()))

    def nodes(self):
        return set(self._adj)
