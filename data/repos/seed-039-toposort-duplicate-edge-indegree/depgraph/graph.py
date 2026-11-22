from collections import defaultdict


class DepGraph:
    """Directed graph; an edge dep -> pkg means pkg depends on dep."""

    def __init__(self):
        self.nodes = set()
        self.successors = defaultdict(set)
        self.indegree = defaultdict(int)

    def add_node(self, name):
        self.nodes.add(name)
        self.indegree[name] += 0

    def add_dependency(self, pkg, dep):
        self.add_node(pkg)
        self.add_node(dep)
        self.successors[dep].add(pkg)
        self.indegree[pkg] += 1

    def dependents(self, name):
        return sorted(self.successors.get(name, ()))
