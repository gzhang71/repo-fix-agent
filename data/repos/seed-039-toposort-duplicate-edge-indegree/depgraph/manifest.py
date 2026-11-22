from .graph import DepGraph
from .toposort import topo_order


def build_graph(manifest):
    """manifest: {package: [dependency, ...]}"""
    g = DepGraph()
    for pkg, deps in manifest.items():
        g.add_node(pkg)
        for dep in deps:
            g.add_dependency(pkg, dep)
    return g


def install_order(manifest):
    return topo_order(build_graph(manifest))
