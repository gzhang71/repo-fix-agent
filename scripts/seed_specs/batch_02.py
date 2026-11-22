"""Seed instances 026-040: data structures & algorithms."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V

SPECS = [
    # 26 ---------------------------------------------------------------------------------------
    dict(
        id="seed-026-heap-priority-tiebreak",
        bug_type="heap-tiebreak", difficulty="medium",
        issue="""\
            JobQueue crashes when two jobs share a priority

            Pushing two jobs with the same priority blows up:

            ```
            >>> from jobs import JobQueue
            >>> from jobs.models import Job
            >>> q = JobQueue()
            >>> q.push(Job("resize"), priority=1)
            >>> q.push(Job("thumbnail"), priority=1)
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File ".../jobs/queue.py", line 11, in push
                heapq.heappush(self._heap, (priority, job))
            TypeError: '<' not supported between instances of 'Job' and 'Job'
            ```

            Same-priority jobs should just run in the order they were submitted.
            """,
        files={
            "jobs/__init__.py": "from .queue import JobQueue\n",
            "jobs/models.py": '''\
                from dataclasses import dataclass, field


                @dataclass
                class Job:
                    name: str
                    payload: dict = field(default_factory=dict)
                ''',
            "jobs/queue.py": '''\
                import heapq


                class JobQueue:
                    """Min-priority queue of jobs: a lower priority number runs first."""

                    def __init__(self):
                        self._heap = []

                    def push(self, job, priority=0):
                        heapq.heappush(self._heap, (priority, job))

                    def pop(self):
                        if not self._heap:
                            raise IndexError("pop from empty JobQueue")
                        return heapq.heappop(self._heap)[1]

                    def __len__(self):
                        return len(self._heap)

                    def drain(self):
                        while self._heap:
                            yield self.pop()
                ''',
            "jobs/worker.py": '''\
                def run_all(queue, handler):
                    """Pop every job in priority order and return the handler results."""
                    return [handler(job) for job in queue.drain()]
                ''',
            "tests/test_queue.py": '''\
                import pytest

                from jobs import JobQueue
                from jobs.models import Job
                from jobs.worker import run_all


                def test_lower_priority_first():
                    q = JobQueue()
                    q.push(Job("c"), priority=3)
                    q.push(Job("a"), priority=1)
                    q.push(Job("b"), priority=2)
                    assert run_all(q, lambda j: j.name) == ["a", "b", "c"]
                    assert len(q) == 0


                def test_pop_empty_raises():
                    with pytest.raises(IndexError):
                        JobQueue().pop()
                ''',
        },
        hidden_test='''\
            from jobs import JobQueue
            from jobs.models import Job
            from jobs.worker import run_all


            def test_equal_priorities_run_fifo():
                q = JobQueue()
                for name in ["resize", "thumbnail", "upload", "notify"]:
                    q.push(Job(name), priority=1)
                assert run_all(q, lambda j: j.name) == ["resize", "thumbnail", "upload", "notify"]


            def test_mixed_priorities_with_ties():
                q = JobQueue()
                q.push(Job("late-1"), priority=5)
                q.push(Job("urgent"), priority=0)
                q.push(Job("late-2"), priority=5)
                q.push(Job("mid"), priority=2)
                assert [q.pop().name for _ in range(4)] == ["urgent", "mid", "late-1", "late-2"]


            def test_distinct_priorities_still_ordered():
                q = JobQueue()
                q.push(Job("b"), priority=2)
                q.push(Job("a"), priority=1)
                assert [j.name for j in q.drain()] == ["a", "b"]
            ''',
        p2p=["tests/test_queue.py::test_lower_priority_first", "tests/test_queue.py::test_pop_empty_raises"],
        steps=[
            ("Find where jobs are pushed onto the heap.", S("heappush|heappop", "jobs")),
            ("Read the queue implementation.", V("jobs/queue.py")),
            ("Reproduce with two same-priority jobs.",
             B("python -c 'from jobs import JobQueue; from jobs.models import Job; q=JobQueue(); "
               "q.push(Job(\"resize\"), 1); q.push(Job(\"thumbnail\"), 1); print([j.name for j in q.drain()])'")),
            ("On a priority tie heapq falls back to comparing the Job objects, which aren't orderable. "
             "Add a monotonically increasing sequence number as a tie-breaker, which also gives FIFO order.",
             R("jobs/queue.py", "import heapq\n", "import heapq\nimport itertools\n")),
            ("Store (priority, seq, job) entries.",
             R("jobs/queue.py",
               "        self._heap = []\n\n    def push(self, job, priority=0):\n"
               "        heapq.heappush(self._heap, (priority, job))\n",
               "        self._heap = []\n        self._seq = itertools.count()\n\n    def push(self, job, priority=0):\n"
               "        heapq.heappush(self._heap, (priority, next(self._seq), job))\n")),
            ("The job is now the third element of the entry.",
             R("jobs/queue.py", "return heapq.heappop(self._heap)[1]", "return heapq.heappop(self._heap)[2]")),
            ("Re-run the repro; expect submission order.",
             B("python -c 'from jobs import JobQueue; from jobs.models import Job; q=JobQueue(); "
               "q.push(Job(\"resize\"), 1); q.push(Job(\"thumbnail\"), 1); q.push(Job(\"urgent\"), 0); "
               "print([j.name for j in q.drain()])'")),
            ("Run the test suite.", T()),
            ("Ties are FIFO and no longer compare Job objects.", SUBMIT),
        ],
    ),
    # 27 ---------------------------------------------------------------------------------------
    dict(
        id="seed-027-grid-bfs-negative-index",
        bug_type="negative-index-wraparound", difficulty="medium",
        issue="""\
            Maze solver walks through the edge of the map

            For this maze the only route from S to E goes down and around the bottom (6 moves):

                S#E
                .#.
                ...

            but `path_length(maze)` returns 4, and `shortest_path(maze)` contains cells like
            `(-1, 0)` and `(-1, 1)` that aren't on the grid at all. Mazes with a full wall border
            work fine; it only happens when the start or path touches the top/left edge.
            """,
        files={
            "maze/__init__.py": "",
            "maze/grid.py": '''\
                WALL = "#"


                def parse(text):
                    """Parse a maze drawn with '#' walls and '.' floor into a list of row strings."""
                    rows = [line.strip() for line in text.strip().splitlines()]
                    if len({len(r) for r in rows}) != 1:
                        raise ValueError("maze rows must all have the same width")
                    return rows


                def find(rows, ch):
                    for r, row in enumerate(rows):
                        c = row.find(ch)
                        if c != -1:
                            return (r, c)
                    raise ValueError(f"{ch!r} not found in maze")


                def neighbors(rows, pos):
                    """Yield open cells orthogonally adjacent to pos."""
                    r, c = pos
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        try:
                            cell = rows[nr][nc]
                        except IndexError:
                            continue
                        if cell != WALL:
                            yield (nr, nc)
                ''',
            "maze/solver.py": '''\
                from collections import deque

                from .grid import find, neighbors, parse


                def shortest_path(text):
                    """Return the list of cells from S to E (inclusive), or None if unreachable."""
                    rows = parse(text)
                    start, goal = find(rows, "S"), find(rows, "E")
                    prev = {start: None}
                    queue = deque([start])
                    while queue:
                        cur = queue.popleft()
                        if cur == goal:
                            path = []
                            while cur is not None:
                                path.append(cur)
                                cur = prev[cur]
                            return path[::-1]
                        for nxt in neighbors(rows, cur):
                            if nxt not in prev:
                                prev[nxt] = cur
                                queue.append(nxt)
                    return None


                def path_length(text):
                    path = shortest_path(text)
                    return None if path is None else len(path) - 1
                ''',
            "tests/test_solver.py": '''\
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
                    assert shortest_path("#####\\n#S#E#\\n#####") is None


                def test_ragged_rows_rejected():
                    with pytest.raises(ValueError):
                        parse("..\\n...")
                ''',
        },
        hidden_test='''\
            from maze.grid import neighbors, parse
            from maze.solver import path_length, shortest_path

            OPEN_EDGES = "S#E\\n.#.\\n..."


            def test_path_does_not_wrap_around_edges():
                assert path_length(OPEN_EDGES) == 6
                path = shortest_path(OPEN_EDGES)
                assert all(0 <= r < 3 and 0 <= c < 3 for r, c in path)


            def test_corner_neighbors_stay_on_grid():
                rows = parse("..\\n..")
                assert sorted(neighbors(rows, (0, 0))) == [(0, 1), (1, 0)]
                assert sorted(neighbors(rows, (1, 1))) == [(0, 1), (1, 0)]


            def test_unreachable_with_open_edges():
                assert shortest_path("S..\\n###\\n..E") is None
            ''',
        p2p=["tests/test_solver.py::test_walled_maze", "tests/test_solver.py::test_unreachable",
             "tests/test_solver.py::test_ragged_rows_rejected"],
        steps=[
            ("Start from the solver entry point.", V("maze/solver.py")),
            ("Reproduce with the maze from the issue.",
             B("python -c 'from maze.solver import shortest_path as sp; print(sp(\"S#E\\n.#.\\n...\"))'")),
            ("The BFS itself looks fine; the bogus (-1, c) cells must come from neighbors(). Look at it.",
             S("def neighbors", "maze")),
            ("Read grid.py.", V("maze/grid.py", 20, 30)),
            ("rows[-1] doesn't raise IndexError in Python, it wraps to the last row, so the try/except "
             "only guards the bottom/right edges. Bounds-check explicitly instead.",
             R("maze/grid.py",
               "        try:\n            cell = rows[nr][nc]\n        except IndexError:\n            continue\n"
               "        if cell != WALL:\n",
               "        if not (0 <= nr < len(rows) and 0 <= nc < len(rows[nr])):\n            continue\n"
               "        if rows[nr][nc] != WALL:\n")),
            ("Re-run the repro; expect the 6-move route around the bottom.",
             B("python -c 'from maze.solver import shortest_path as sp, path_length as pl; m=\"S#E\\n.#.\\n...\"; "
               "print(sp(m), pl(m))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Neighbors are now bounded on all four edges.", SUBMIT),
        ],
    ),
    # 28 ---------------------------------------------------------------------------------------
    dict(
        id="seed-028-dijkstra-visited-on-push",
        bug_type="algorithm-invariant", difficulty="medium",
        issue="""\
            route() picks an expensive direct edge over a cheaper multi-hop path

            With edges A-D (10), A-B (1), B-C (1), C-D (1):

                g = Graph()
                g.add_edge("A", "D", 10); g.add_edge("A", "B", 1)
                g.add_edge("B", "C", 1); g.add_edge("C", "D", 1)
                route(g, "A", "D")   # (['A', 'D'], 10)

            Expected `(['A', 'B', 'C', 'D'], 3)`. Seems like whichever path reaches a node first
            wins, regardless of cost.
            """,
        files={
            "routing/__init__.py": "",
            "routing/graph.py": '''\
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
                ''',
            "routing/dijkstra.py": '''\
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
                ''',
            "tests/test_routing.py": '''\
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
                ''',
        },
        hidden_test='''\
            from routing.dijkstra import route, shortest_paths
            from routing.graph import Graph


            def test_cheaper_multi_hop_beats_direct_edge():
                g = Graph()
                g.add_edge("A", "D", 10)
                g.add_edge("A", "B", 1)
                g.add_edge("B", "C", 1)
                g.add_edge("C", "D", 1)
                assert route(g, "A", "D") == (["A", "B", "C", "D"], 3)


            def test_all_distances_optimal():
                g = Graph()
                for u, v, w in [("s", "a", 4), ("s", "b", 1), ("b", "a", 2), ("a", "t", 1), ("b", "t", 5)]:
                    g.add_edge(u, v, w, bidirectional=False)
                dist, _ = shortest_paths(g, "s")
                assert dist == {"s": 0, "a": 3, "b": 1, "t": 4}


            def test_source_to_itself():
                g = Graph()
                g.add_edge("A", "B", 1)
                assert route(g, "A", "A") == (["A"], 0)
            ''',
        p2p=["tests/test_routing.py::test_line_graph", "tests/test_routing.py::test_unreachable",
             "tests/test_routing.py::test_negative_weight_rejected"],
        steps=[
            ("See what the routing package contains.", F("routing/*.py")),
            ("Read the Dijkstra implementation.", V("routing/dijkstra.py")),
            ("Reproduce the example from the issue.",
             B("python -c 'from routing.graph import Graph; from routing.dijkstra import route; g=Graph(); "
               "[g.add_edge(u,v,w) for u,v,w in [(\"A\",\"D\",10),(\"A\",\"B\",1),(\"B\",\"C\",1),(\"C\",\"D\",1)]]; "
               "print(route(g, \"A\", \"D\"))'")),
            ("Nodes are marked visited when first discovered, so a node's first tentative distance can never be "
             "improved. In Dijkstra a node is final only when it is popped. Start with an empty visited set.",
             R("routing/dijkstra.py", "    visited = {source}\n", "    visited = set()\n")),
            ("Mark nodes visited on pop (skipping stale heap entries) and always allow relaxing unvisited neighbors.",
             R("routing/dijkstra.py",
               "        d, node = heapq.heappop(heap)\n        for nbr, w in graph.neighbors(node):\n"
               "            if nbr in visited:\n                continue\n            visited.add(nbr)\n",
               "        d, node = heapq.heappop(heap)\n        if node in visited:\n            continue\n"
               "        visited.add(node)\n        for nbr, w in graph.neighbors(node):\n"
               "            if nbr in visited:\n                continue\n")),
            ("Re-run the repro.",
             B("python -c 'from routing.graph import Graph; from routing.dijkstra import route; g=Graph(); "
               "[g.add_edge(u,v,w) for u,v,w in [(\"A\",\"D\",10),(\"A\",\"B\",1),(\"B\",\"C\",1),(\"C\",\"D\",1)]]; "
               "print(route(g, \"A\", \"D\"))'")),
            ("Run the tests.", T()),
            ("Dijkstra now finalizes nodes on pop.", SUBMIT),
        ],
    ),
    # 29 ---------------------------------------------------------------------------------------
    dict(
        id="seed-029-interval-merge-contained",
        bug_type="interval-merge", difficulty="medium",
        issue="""\
            free_slots shows me as available during an all-afternoon block

            I have an offsite 13:00-18:00 and a 1:1 at 14:00-14:30 (which overlaps it). Times are
            minutes since midnight:

                >>> free_slots([(780, 1080), (840, 870)], day_start=540, day_end=1080)
                [(540, 780), (870, 1080)]

            Expected only `[(540, 780)]` - I'm busy from 13:00 until the end of the day. Without
            the 1:1 the result is correct.
            """,
        files={
            "scheduling/__init__.py": "",
            "scheduling/intervals.py": '''\
                def merge(intervals):
                    """Merge overlapping or touching (start, end) intervals. Returns a sorted list."""
                    merged = []
                    for start, end in sorted(intervals):
                        if merged and start <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], end)
                        else:
                            merged.append((start, end))
                    return merged


                def total_length(intervals):
                    """Total time covered by the intervals, counting overlaps once."""
                    return sum(end - start for start, end in merge(intervals))
                ''',
            "scheduling/availability.py": '''\
                from .intervals import merge


                def free_slots(busy, day_start, day_end, min_length=0):
                    """Return the gaps between busy intervals within [day_start, day_end]."""
                    slots = []
                    cursor = day_start
                    for start, end in merge(busy):
                        if end <= day_start or start >= day_end:
                            continue
                        if start > cursor and start - cursor >= min_length:
                            slots.append((cursor, start))
                        cursor = max(cursor, end)
                    if day_end > cursor and day_end - cursor >= min_length:
                        slots.append((cursor, day_end))
                    return slots
                ''',
            "tests/test_scheduling.py": '''\
                from scheduling.availability import free_slots
                from scheduling.intervals import merge


                def test_merge_overlapping():
                    assert merge([(5, 8), (1, 3), (2, 4)]) == [(1, 4), (5, 8)]


                def test_merge_touching():
                    assert merge([(1, 2), (2, 3)]) == [(1, 3)]


                def test_free_slots_basic():
                    busy = [(600, 660), (720, 780)]
                    assert free_slots(busy, 540, 1020) == [(540, 600), (660, 720), (780, 1020)]


                def test_free_slots_min_length():
                    busy = [(600, 610)]
                    assert free_slots(busy, 540, 1020, min_length=480) == []
                ''',
        },
        hidden_test='''\
            from scheduling.availability import free_slots
            from scheduling.intervals import merge, total_length


            def test_merge_keeps_end_of_containing_interval():
                assert merge([(1, 10), (2, 3)]) == [(1, 10)]
                assert merge([(1, 10), (2, 3), (4, 12)]) == [(1, 12)]


            def test_free_slots_with_nested_meeting():
                assert free_slots([(780, 1080), (840, 870)], day_start=540, day_end=1080) == [(540, 780)]


            def test_total_length_counts_overlap_once():
                assert total_length([(0, 10), (2, 5), (8, 12)]) == 12
            ''',
        p2p=["tests/test_scheduling.py::test_merge_overlapping", "tests/test_scheduling.py::test_merge_touching",
             "tests/test_scheduling.py::test_free_slots_basic", "tests/test_scheduling.py::test_free_slots_min_length"],
        steps=[
            ("Find free_slots.", S("def free_slots|def merge", "scheduling")),
            ("Read free_slots.", V("scheduling/availability.py")),
            ("Reproduce the issue's call.",
             B("python -c 'from scheduling.availability import free_slots; print(free_slots([(780,1080),(840,870)], 540, 1080))'")),
            ("The gap logic looks right, so check what merge() hands it.",
             B("python -c 'from scheduling.intervals import merge; print(merge([(780,1080),(840,870)]))'")),
            ("merge() shrank the offsite to end at 870. Look at it.", V("scheduling/intervals.py", 1, 9)),
            ("When the next interval is fully contained, the merged end is overwritten with the smaller end. "
             "Keep the max of both ends.",
             R("scheduling/intervals.py", "merged[-1] = (merged[-1][0], end)",
               "merged[-1] = (merged[-1][0], max(merged[-1][1], end))")),
            ("Re-run both repros.",
             B("python -c 'from scheduling.intervals import merge; from scheduling.availability import free_slots; "
               "print(merge([(780,1080),(840,870)]), free_slots([(780,1080),(840,870)], 540, 1080))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Fixed at the source in merge(); total_length benefits too.", SUBMIT),
        ],
    ),
    # 30 ---------------------------------------------------------------------------------------
    dict(
        id="seed-030-union-find-size-root",
        bug_type="union-find", difficulty="medium",
        issue="""\
            largest_cluster_size always says 1

            `clusters()` groups the hosts correctly, but `largest_cluster_size()` on the same input
            returns 1:

                hosts = ["a", "b", "c", "d", "e"]
                links = [("a", "b"), ("b", "c"), ("c", "d")]
                clusters(hosts, links)              # [['a', 'b', 'c', 'd'], ['e']]
                largest_cluster_size(hosts, links)  # 1, expected 4

            Our capacity alerting relies on this number.
            """,
        files={
            "netgraph/__init__.py": "",
            "netgraph/dsu.py": '''\
                class DisjointSet:
                    """Union-find with path compression and union by size."""

                    def __init__(self):
                        self.parent = {}
                        self.size = {}

                    def add(self, x):
                        if x not in self.parent:
                            self.parent[x] = x
                            self.size[x] = 1

                    def find(self, x):
                        self.add(x)
                        root = x
                        while self.parent[root] != root:
                            root = self.parent[root]
                        while self.parent[x] != root:
                            self.parent[x], x = root, self.parent[x]
                        return root

                    def union(self, a, b):
                        """Join the sets containing a and b. Returns False if already joined."""
                        ra, rb = self.find(a), self.find(b)
                        if ra == rb:
                            return False
                        if self.size[ra] < self.size[rb]:
                            ra, rb = rb, ra
                        self.parent[ra] = rb
                        self.size[ra] += self.size[rb]
                        return True

                    def connected(self, a, b):
                        return self.find(a) == self.find(b)

                    def component_size(self, x):
                        return self.size[self.find(x)]
                ''',
            "netgraph/clusters.py": '''\
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
                ''',
            "tests/test_clusters.py": '''\
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
                ''',
        },
        hidden_test='''\
            from netgraph.clusters import largest_cluster_size
            from netgraph.dsu import DisjointSet


            def test_largest_cluster_size():
                hosts = ["a", "b", "c", "d", "e"]
                assert largest_cluster_size(hosts, [("a", "b"), ("b", "c"), ("c", "d")]) == 4


            def test_component_size_after_unions():
                ds = DisjointSet()
                ds.union(1, 2)
                ds.union(3, 4)
                ds.union(5, 4)
                assert ds.component_size(1) == 2
                assert ds.component_size(5) == 3
                ds.union(2, 3)
                assert all(ds.component_size(x) == 5 for x in range(1, 6))


            def test_union_already_joined():
                ds = DisjointSet()
                assert ds.union("x", "y") is True
                assert ds.union("y", "x") is False
                assert ds.connected("x", "y")
            ''',
        p2p=["tests/test_clusters.py::test_clusters", "tests/test_clusters.py::test_connected",
             "tests/test_clusters.py::test_largest_cluster_empty"],
        steps=[
            ("Look at how the two functions differ.", V("netgraph/clusters.py")),
            ("Reproduce.",
             B("python -c 'from netgraph.clusters import clusters, largest_cluster_size as L; h=list(\"abcde\"); "
               "l=[(\"a\",\"b\"),(\"b\",\"c\"),(\"c\",\"d\")]; print(clusters(h,l), L(h,l))'")),
            ("clusters() only uses find() while largest_cluster_size() relies on component_size(). Check the DSU.",
             V("netgraph/dsu.py", 22, 36)),
            ("Inspect parent/size after a couple of unions.",
             B("python -c 'from netgraph.dsu import DisjointSet; ds=DisjointSet(); ds.union(1,2); ds.union(2,3); "
               "print(ds.parent, ds.size, ds.find(1))'")),
            ("After the swap `ra` is the larger root and gets the size added, but it is attached under `rb`, so "
             "the root's size is never updated. Attach the smaller root under the larger one.",
             R("netgraph/dsu.py", "        self.parent[ra] = rb\n", "        self.parent[rb] = ra\n")),
            ("Re-run the repro.",
             B("python -c 'from netgraph.clusters import clusters, largest_cluster_size as L; h=list(\"abcde\"); "
               "l=[(\"a\",\"b\"),(\"b\",\"c\"),(\"c\",\"d\")]; print(clusters(h,l), L(h,l))'")),
            ("Run the tests.", T()),
            ("Union-by-size now keeps sizes on the actual root.", SUBMIT),
        ],
    ),
    # 31 ---------------------------------------------------------------------------------------
    dict(
        id="seed-031-trie-prefix-word-terminal",
        bug_type="state-update", difficulty="easy",
        issue="""\
            Autocomplete loses words that are a prefix of an earlier word

            If "banner" is added before "ban", then "ban" never shows up:

                trie = Trie(["banner", "ban"])
                print("ban" in trie)          # False
                print(suggest(trie, "ba"))    # ['banner']
                print(len(trie))              # 1

            Adding them in the opposite order works.
            """,
        files={
            "autocomplete/__init__.py": "from .trie import Trie\n",
            "autocomplete/trie.py": '''\
                class _Node:
                    __slots__ = ("children", "terminal")

                    def __init__(self):
                        self.children = {}
                        self.terminal = False


                class Trie:
                    def __init__(self, words=()):
                        self.root = _Node()
                        self._count = 0
                        for w in words:
                            self.insert(w)

                    def insert(self, word):
                        node = self.root
                        created = False
                        for ch in word:
                            if ch not in node.children:
                                node.children[ch] = _Node()
                                created = True
                            node = node.children[ch]
                        if created:  # only count genuinely new words
                            node.terminal = True
                            self._count += 1

                    def _walk(self, prefix):
                        node = self.root
                        for ch in prefix:
                            node = node.children.get(ch)
                            if node is None:
                                return None
                        return node

                    def __contains__(self, word):
                        node = self._walk(word)
                        return node is not None and node.terminal

                    def __len__(self):
                        return self._count

                    def words_with_prefix(self, prefix):
                        start = self._walk(prefix)
                        out = []
                        stack = [(start, prefix)] if start else []
                        while stack:
                            node, sofar = stack.pop()
                            if node.terminal:
                                out.append(sofar)
                            for ch, child in node.children.items():
                                stack.append((child, sofar + ch))
                        return sorted(out)
                ''',
            "autocomplete/suggest.py": '''\
                def suggest(trie, prefix, limit=5):
                    """Up to `limit` completions for prefix, shortest first then alphabetical."""
                    words = trie.words_with_prefix(prefix.lower())
                    return sorted(words, key=lambda w: (len(w), w))[:limit]
                ''',
            "tests/test_trie.py": '''\
                from autocomplete import Trie
                from autocomplete.suggest import suggest


                def test_contains_and_len():
                    t = Trie(["car", "cart", "dog"])
                    assert "cart" in t and "car" in t
                    assert "ca" not in t
                    assert len(t) == 3


                def test_suggest_order_and_limit():
                    t = Trie(["car", "card", "care", "cat", "dog"])
                    assert suggest(t, "Ca", limit=3) == ["car", "cat", "card"]
                ''',
        },
        hidden_test='''\
            from autocomplete import Trie
            from autocomplete.suggest import suggest


            def test_prefix_word_inserted_after_longer_word():
                t = Trie(["cart", "car"])
                assert "car" in t
                assert len(t) == 2
                assert suggest(t, "ca") == ["car", "cart"]


            def test_duplicate_insert_counted_once():
                t = Trie(["car", "car", "cart", "cart"])
                assert len(t) == 2
            ''',
        p2p=["tests/test_trie.py::test_contains_and_len", "tests/test_trie.py::test_suggest_order_and_limit"],
        steps=[
            ("Locate the trie.", F("*.py")),
            ("Read insert().", V("autocomplete/trie.py", 9, 25)),
            ("Reproduce the order dependence.",
             B("python -c 'from autocomplete import Trie; t=Trie([\"cart\",\"car\"]); print(\"car\" in t, len(t))'")),
            ("`created` is used as a proxy for 'new word', but inserting a prefix of an existing word creates no "
             "nodes. The real check is whether the final node is already terminal.",
             R("autocomplete/trie.py",
               "        if created:  # only count genuinely new words\n",
               "        if not node.terminal:  # only count genuinely new words\n")),
            ("`created` is now unused; drop it.",
             R("autocomplete/trie.py",
               "        created = False\n        for ch in word:\n            if ch not in node.children:\n"
               "                node.children[ch] = _Node()\n                created = True\n",
               "        for ch in word:\n            if ch not in node.children:\n"
               "                node.children[ch] = _Node()\n")),
            ("Verify both orders and duplicates.",
             B("python -c 'from autocomplete import Trie; a=Trie([\"cart\",\"car\"]); b=Trie([\"car\",\"car\",\"cart\"]); "
               "print(\"car\" in a, len(a), len(b))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 32 ---------------------------------------------------------------------------------------
    dict(
        id="seed-032-multikey-sort-order",
        bug_type="sort-stability", difficulty="medium",
        issue="""\
            Department report isn't grouped by department

            The "top earners by department" report is supposed to list each department together
            (alphabetically), highest salary first within each. Instead the rows come out ordered
            by salary across the whole company, with departments interleaved. Sorting by a single
            field works fine.
            """,
        files={
            "records/__init__.py": "",
            "records/sorting.py": '''\
                def sort_by(rows, *fields):
                    """Sort dict rows by several fields. Prefix a field with '-' for descending.

                    sort_by(rows, "dept", "-salary") sorts by dept ascending, then by salary
                    descending within each dept. The sort is stable.
                    """
                    result = list(rows)
                    for spec in fields:
                        name = spec.lstrip("-")
                        result.sort(key=lambda r: r[name], reverse=spec.startswith("-"))
                    return result
                ''',
            "records/report.py": '''\
                from .sorting import sort_by


                def top_earners_by_dept(rows):
                    """Lines like 'eng  Ana  120000', grouped by department, highest salary first."""
                    return [f"{r['dept']:<5}{r['name']:<6}{r['salary']}" for r in sort_by(rows, "dept", "-salary")]
                ''',
            "tests/test_sorting.py": '''\
                from records.sorting import sort_by

                ROWS = [
                    {"name": "Ana", "dept": "eng", "salary": 120},
                    {"name": "Bo", "dept": "ops", "salary": 90},
                    {"name": "Cy", "dept": "eng", "salary": 100},
                ]


                def test_single_field_ascending():
                    assert [r["name"] for r in sort_by(ROWS, "salary")] == ["Bo", "Cy", "Ana"]


                def test_single_field_descending():
                    assert [r["name"] for r in sort_by(ROWS, "-salary")] == ["Ana", "Cy", "Bo"]


                def test_stable_for_equal_keys():
                    assert [r["name"] for r in sort_by(ROWS, "dept")] == ["Ana", "Cy", "Bo"]


                def test_input_not_mutated():
                    rows = list(ROWS)
                    sort_by(rows, "salary")
                    assert rows == ROWS
                ''',
        },
        hidden_test='''\
            from records.report import top_earners_by_dept
            from records.sorting import sort_by

            ROWS = [
                {"name": "Dee", "dept": "ops", "salary": 130},
                {"name": "Ana", "dept": "eng", "salary": 120},
                {"name": "Bo", "dept": "ops", "salary": 90},
                {"name": "Cy", "dept": "eng", "salary": 100},
                {"name": "Eve", "dept": "eng", "salary": 140},
            ]


            def test_first_field_is_primary_key():
                out = sort_by(ROWS, "dept", "-salary")
                assert [r["name"] for r in out] == ["Eve", "Ana", "Cy", "Dee", "Bo"]


            def test_three_fields():
                rows = [{"a": 1, "b": 2, "c": 1}, {"a": 1, "b": 1, "c": 2}, {"a": 0, "b": 2, "c": 3},
                        {"a": 1, "b": 1, "c": 1}]
                out = sort_by(rows, "-a", "b", "-c")
                assert [(r["a"], r["b"], r["c"]) for r in out] == [(1, 1, 2), (1, 1, 1), (1, 2, 1), (0, 2, 3)]


            def test_report_grouped_by_dept():
                lines = top_earners_by_dept(ROWS)
                assert [l.split()[0] for l in lines] == ["eng", "eng", "eng", "ops", "ops"]
                assert lines[0].split()[1] == "Eve"
            ''',
        p2p=["tests/test_sorting.py::test_single_field_ascending", "tests/test_sorting.py::test_single_field_descending",
             "tests/test_sorting.py::test_stable_for_equal_keys", "tests/test_sorting.py::test_input_not_mutated"],
        steps=[
            ("Find the report.", S("top_earners_by_dept|def sort_by", "records")),
            ("The report delegates ordering to sort_by(rows, 'dept', '-salary'). Read sort_by.",
             V("records/sorting.py")),
            ("Reproduce with a few rows.",
             B("python -c 'from records.sorting import sort_by; rows=[{\"d\":\"ops\",\"s\":130},{\"d\":\"eng\",\"s\":120},"
               "{\"d\":\"ops\",\"s\":90},{\"d\":\"eng\",\"s\":140}]; print([(r[\"d\"], r[\"s\"]) for r in sort_by(rows, \"d\", \"-s\")])'")),
            ("Chaining stable sorts only works if you sort by the least significant key first. Iterating fields "
             "in order makes the last field the primary key. Apply them in reverse.",
             R("records/sorting.py", "    for spec in fields:\n", "    for spec in reversed(fields):\n")),
            ("Re-run the repro; expect eng rows first, salaries descending within each dept.",
             B("python -c 'from records.sorting import sort_by; rows=[{\"d\":\"ops\",\"s\":130},{\"d\":\"eng\",\"s\":120},"
               "{\"d\":\"ops\",\"s\":90},{\"d\":\"eng\",\"s\":140}]; print([(r[\"d\"], r[\"s\"]) for r in sort_by(rows, \"d\", \"-s\")])'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Fixed.", SUBMIT),
        ],
    ),
    # 33 ---------------------------------------------------------------------------------------
    dict(
        id="seed-033-ring-buffer-iter-order",
        bug_type="ring-buffer-wraparound", difficulty="medium",
        issue="""\
            `tail` returns lines out of order for longer logs

            ```
            >>> from logtail.tail import tail
            >>> tail([f"line {i}" for i in range(1, 13)], n=5)
            ['line 11', 'line 12', 'line 8', 'line 9', 'line 10']
            ```

            It has the right five lines, but not in order. Short logs (fewer than n lines) come
            out fine.
            """,
        files={
            "logtail/__init__.py": "",
            "logtail/ringbuffer.py": '''\
                class RingBuffer:
                    """Fixed-capacity buffer that keeps the most recent `capacity` items."""

                    def __init__(self, capacity):
                        if capacity < 1:
                            raise ValueError("capacity must be >= 1")
                        self.capacity = capacity
                        self._items = [None] * capacity
                        self._start = 0
                        self._size = 0

                    def append(self, item):
                        end = (self._start + self._size) % self.capacity
                        self._items[end] = item
                        if self._size < self.capacity:
                            self._size += 1
                        else:
                            self._start = (self._start + 1) % self.capacity

                    def latest(self):
                        if not self._size:
                            raise IndexError("empty buffer")
                        return self._items[(self._start + self._size - 1) % self.capacity]

                    def __len__(self):
                        return self._size

                    def __iter__(self):
                        for i in range(self._size):
                            yield self._items[i]
                ''',
            "logtail/tail.py": '''\
                from .ringbuffer import RingBuffer


                def tail(lines, n=10):
                    """Last n lines, oldest first."""
                    buf = RingBuffer(n)
                    for line in lines:
                        buf.append(line.rstrip("\\n"))
                    return list(buf)


                def tail_matching(lines, needle, n=10):
                    return tail((l for l in lines if needle in l), n)
                ''',
            "tests/test_tail.py": '''\
                import pytest

                from logtail.ringbuffer import RingBuffer
                from logtail.tail import tail, tail_matching


                def test_short_log():
                    assert tail(["a\\n", "b\\n"], n=5) == ["a", "b"]


                def test_latest_after_wrap():
                    buf = RingBuffer(3)
                    for i in range(7):
                        buf.append(i)
                    assert buf.latest() == 6
                    assert len(buf) == 3


                def test_tail_matching_short():
                    assert tail_matching(["ok", "ERR 1", "ok", "ERR 2"], "ERR", n=5) == ["ERR 1", "ERR 2"]


                def test_bad_capacity():
                    with pytest.raises(ValueError):
                        RingBuffer(0)
                ''',
        },
        hidden_test='''\
            from logtail.ringbuffer import RingBuffer
            from logtail.tail import tail


            def test_tail_keeps_order_after_wrap():
                lines = [f"line {i}" for i in range(1, 13)]
                assert tail(lines, n=5) == ["line 8", "line 9", "line 10", "line 11", "line 12"]


            def test_iter_oldest_first_after_many_wraps():
                buf = RingBuffer(3)
                for i in range(10):
                    buf.append(i)
                assert list(buf) == [7, 8, 9]


            def test_exactly_full():
                buf = RingBuffer(3)
                for i in range(3):
                    buf.append(i)
                assert list(buf) == [0, 1, 2]
            ''',
        p2p=["tests/test_tail.py::test_short_log", "tests/test_tail.py::test_latest_after_wrap",
             "tests/test_tail.py::test_tail_matching_short", "tests/test_tail.py::test_bad_capacity"],
        steps=[
            ("Read tail().", V("logtail/tail.py")),
            ("Reproduce the issue's example.",
             B("python -c 'from logtail.tail import tail; print(tail([f\"line {i}\" for i in range(1, 13)], n=5))'")),
            ("tail() just lists the RingBuffer, so the ordering comes from its iterator.", V("logtail/ringbuffer.py")),
            ("append() advances `_start` once the buffer is full and latest() accounts for it, but __iter__ reads "
             "physical slots 0..size-1. Iterate from the logical start instead.",
             R("logtail/ringbuffer.py", "            yield self._items[i]\n",
               "            yield self._items[(self._start + i) % self.capacity]\n")),
            ("Re-run the repro.",
             B("python -c 'from logtail.tail import tail; print(tail([f\"line {i}\" for i in range(1, 13)], n=5))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Iteration now starts at the oldest element.", SUBMIT),
        ],
    ),
    # 34 ---------------------------------------------------------------------------------------
    dict(
        id="seed-034-token-bucket-refill-cap",
        bug_type="missing-clamp", difficulty="easy",
        issue="""\
            Rate limiter allows huge bursts after a client goes idle

            Limiter is configured with capacity=5, refill 1 token/s. A client that was quiet for
            a minute can then fire 60+ requests back-to-back and all of them are allowed. The
            burst should never exceed the bucket capacity.
            """,
        files={
            "ratelimit/__init__.py": "",
            "ratelimit/clock.py": '''\
                class FakeClock:
                    """Manually advanced clock for tests and simulations."""

                    def __init__(self, start=0.0):
                        self.now = start

                    def __call__(self):
                        return self.now

                    def advance(self, seconds):
                        self.now += seconds
                ''',
            "ratelimit/bucket.py": '''\
                class TokenBucket:
                    def __init__(self, capacity, refill_rate, clock):
                        self.capacity = capacity
                        self.refill_rate = refill_rate  # tokens per second
                        self._clock = clock
                        self._tokens = float(capacity)
                        self._last = clock()

                    def _refill(self):
                        now = self._clock()
                        elapsed = now - self._last
                        self._last = now
                        self._tokens = self._tokens + elapsed * self.refill_rate

                    def try_acquire(self, n=1):
                        self._refill()
                        if self._tokens >= n:
                            self._tokens -= n
                            return True
                        return False

                    @property
                    def available(self):
                        self._refill()
                        return self._tokens
                ''',
            "ratelimit/limiter.py": '''\
                from .bucket import TokenBucket


                class RateLimiter:
                    """One token bucket per client key."""

                    def __init__(self, capacity, refill_rate, clock):
                        self._args = (capacity, refill_rate, clock)
                        self._buckets = {}

                    def allow(self, key):
                        if key not in self._buckets:
                            self._buckets[key] = TokenBucket(*self._args)
                        return self._buckets[key].try_acquire()
                ''',
            "tests/test_limiter.py": '''\
                from ratelimit.clock import FakeClock
                from ratelimit.limiter import RateLimiter


                def test_initial_burst_limited():
                    lim = RateLimiter(3, 1.0, FakeClock())
                    assert [lim.allow("k") for _ in range(4)] == [True, True, True, False]


                def test_refills_over_time():
                    clock = FakeClock()
                    lim = RateLimiter(2, 1.0, clock)
                    lim.allow("k"); lim.allow("k")
                    assert not lim.allow("k")
                    clock.advance(1.0)
                    assert lim.allow("k")


                def test_keys_independent():
                    lim = RateLimiter(1, 1.0, FakeClock())
                    assert lim.allow("a") and lim.allow("b")
                    assert not lim.allow("a")
                ''',
        },
        hidden_test='''\
            from ratelimit.bucket import TokenBucket
            from ratelimit.clock import FakeClock
            from ratelimit.limiter import RateLimiter


            def test_burst_capped_after_idle():
                clock = FakeClock()
                lim = RateLimiter(5, 1.0, clock)
                lim.allow("k")
                clock.advance(60)
                assert sum(lim.allow("k") for _ in range(100)) == 5


            def test_available_never_exceeds_capacity():
                clock = FakeClock()
                b = TokenBucket(4, 2.0, clock)
                clock.advance(1000)
                assert b.available == 4


            def test_partial_refill():
                clock = FakeClock()
                b = TokenBucket(4, 2.0, clock)
                for _ in range(4):
                    assert b.try_acquire()
                clock.advance(0.5)
                assert b.available == 1
            ''',
        p2p=["tests/test_limiter.py::test_initial_burst_limited", "tests/test_limiter.py::test_refills_over_time",
             "tests/test_limiter.py::test_keys_independent"],
        steps=[
            ("Find the token bucket.", S("class TokenBucket|def _refill")),
            ("Read the refill logic.", V("ratelimit/bucket.py")),
            ("Reproduce with the fake clock: one request, idle 60s, then count how many are allowed.",
             B("python -c 'from ratelimit.clock import FakeClock; from ratelimit.limiter import RateLimiter; "
               "c=FakeClock(); l=RateLimiter(5, 1.0, c); l.allow(\"k\"); c.advance(60); print(sum(l.allow(\"k\") for _ in range(100)))'")),
            ("Tokens accumulate without bound while idle. Clamp the refilled amount to the capacity.",
             R("ratelimit/bucket.py",
               "self._tokens = self._tokens + elapsed * self.refill_rate",
               "self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)")),
            ("Re-run; expect 5.",
             B("python -c 'from ratelimit.clock import FakeClock; from ratelimit.limiter import RateLimiter; "
               "c=FakeClock(); l=RateLimiter(5, 1.0, c); l.allow(\"k\"); c.advance(60); print(sum(l.allow(\"k\") for _ in range(100)))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 35 ---------------------------------------------------------------------------------------
    dict(
        id="seed-035-matrix-transpose-nonsquare",
        bug_type="shape-assumption", difficulty="easy",
        issue="""\
            matmul fails for non-square matrices

            ```
            >>> from matrix.linalg import matmul
            >>> matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
            Traceback (most recent call last):
              File "<string>", line 1, in <module>
              File ".../matrix/linalg.py", line 9, in matmul
                bt = transpose(b)
                     ^^^^^^^^^^^^
              File ".../matrix/core.py", line 7, in transpose
                return [[m[i][j] for i in range(n)] for j in range(n)]
                         ~~~~^^^
            IndexError: list index out of range
            ```

            Expected `[[58, 64], [139, 154]]`. Square matrices work.
            """,
        files={
            "matrix/__init__.py": "",
            "matrix/core.py": '''\
                def shape(m):
                    return (len(m), len(m[0]) if m else 0)


                def transpose(m):
                    n = len(m)
                    return [[m[i][j] for i in range(n)] for j in range(n)]


                def identity(n):
                    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
                ''',
            "matrix/linalg.py": '''\
                from .core import shape, transpose


                def matmul(a, b):
                    ra, ca = shape(a)
                    rb, cb = shape(b)
                    if ca != rb:
                        raise ValueError(f"cannot multiply {ra}x{ca} by {rb}x{cb}")
                    bt = transpose(b)
                    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


                def trace(m):
                    return sum(m[i][i] for i in range(len(m)))
                ''',
            "tests/test_matrix.py": '''\
                import pytest

                from matrix.core import identity, shape, transpose
                from matrix.linalg import matmul, trace


                def test_transpose_square():
                    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


                def test_matmul_identity():
                    m = [[1, 2], [3, 4]]
                    assert matmul(m, identity(2)) == m


                def test_shape_mismatch():
                    with pytest.raises(ValueError):
                        matmul([[1, 2]], [[1, 2]])


                def test_trace_and_shape():
                    assert trace(identity(3)) == 3
                    assert shape([[1, 2, 3]]) == (1, 3)
                ''',
        },
        hidden_test='''\
            from matrix.core import transpose
            from matrix.linalg import matmul


            def test_transpose_rectangular():
                assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
                assert transpose([[1], [2], [3]]) == [[1, 2, 3]]


            def test_matmul_rectangular():
                assert matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]]
                assert matmul([[1, 2]], [[1, 2, 3], [4, 5, 6]]) == [[9, 12, 15]]


            def test_transpose_empty():
                assert transpose([]) == []
            ''',
        p2p=["tests/test_matrix.py::test_transpose_square", "tests/test_matrix.py::test_matmul_identity",
             "tests/test_matrix.py::test_shape_mismatch", "tests/test_matrix.py::test_trace_and_shape"],
        steps=[
            ("Reproduce the traceback from the issue.",
             B("python -c 'from matrix.linalg import matmul; print(matmul([[1,2,3],[4,5,6]], [[7,8],[9,10],[11,12]]))'")),
            ("The error is raised in transpose(), called from matmul. Read core.py.", V("matrix/core.py")),
            ("transpose() uses the row count for both dimensions, so it crashes on tall matrices and silently "
             "truncates wide ones. Build columns with zip, which also handles the empty matrix.",
             R("matrix/core.py",
               "    n = len(m)\n    return [[m[i][j] for i in range(n)] for j in range(n)]\n",
               "    return [list(col) for col in zip(*m)]\n")),
            ("Verify the issue's example plus a wide matrix and an empty one.",
             B("python -c 'from matrix.linalg import matmul; from matrix.core import transpose as t; "
               "print(matmul([[1,2,3],[4,5,6]], [[7,8],[9,10],[11,12]]), t([[1,2,3],[4,5,6]]), t([]))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 36 ---------------------------------------------------------------------------------------
    dict(
        id="seed-036-dedupe-key-not-tracked",
        bug_type="wrong-variable", difficulty="easy",
        issue="""\
            import_contacts crashes: unhashable type: 'dict'

            ```
            >>> from contacts.importer import import_contacts
            >>> import_contacts([{"email": "a@x.io"}, {"email": "A@x.io "}])
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File ".../contacts/importer.py", line 10, in import_contacts
                return list(dedupe(rows, key=normalize_email))
              File ".../contacts/iterutils.py", line 8, in dedupe
                seen.add(item)
            TypeError: unhashable type: 'dict'
            ```

            Contacts should be de-duplicated by normalized email, keeping the first one.
            """,
        files={
            "contacts/__init__.py": "",
            "contacts/iterutils.py": '''\
                def dedupe(items, key=None):
                    """Yield items in order, skipping any whose key (default: the item) was already seen."""
                    seen = set()
                    for item in items:
                        k = item if key is None else key(item)
                        if k in seen:
                            continue
                        seen.add(item)
                        yield item


                def chunked(items, size):
                    batch = []
                    for item in items:
                        batch.append(item)
                        if len(batch) == size:
                            yield batch
                            batch = []
                    if batch:
                        yield batch
                ''',
            "contacts/importer.py": '''\
                from .iterutils import chunked, dedupe


                def normalize_email(row):
                    return row["email"].strip().lower()


                def import_contacts(rows):
                    """Contacts de-duplicated by normalized email; the first occurrence wins."""
                    return list(dedupe(rows, key=normalize_email))


                def batches(rows, size=100):
                    return list(chunked(import_contacts(rows), size))
                ''',
            "tests/test_iterutils.py": '''\
                from contacts.iterutils import chunked, dedupe


                def test_dedupe_plain_values():
                    assert list(dedupe([3, 1, 3, 2, 1])) == [3, 1, 2]


                def test_chunked():
                    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
                ''',
        },
        hidden_test='''\
            from contacts.importer import batches, import_contacts
            from contacts.iterutils import dedupe


            def test_dedupe_with_key():
                assert list(dedupe(["a", "B", "A", "b", "c"], key=str.lower)) == ["a", "B", "c"]


            def test_import_contacts_by_normalized_email():
                rows = [{"email": "a@x.io", "n": 1}, {"email": "A@x.io ", "n": 2}, {"email": "b@x.io", "n": 3}]
                assert [r["n"] for r in import_contacts(rows)] == [1, 3]
                assert batches(rows, size=1) == [[rows[0]], [rows[2]]]


            def test_dedupe_without_key_unchanged():
                assert list(dedupe("abracadabra")) == ["a", "b", "r", "c", "d"]
            ''',
        p2p=["tests/test_iterutils.py::test_dedupe_plain_values", "tests/test_iterutils.py::test_chunked"],
        steps=[
            ("Reproduce the crash.",
             B("python -c 'from contacts.importer import import_contacts; "
               "print(import_contacts([{\"email\": \"a@x.io\"}, {\"email\": \"A@x.io \"}]))' 2>&1 | tail -4")),
            ("The importer just calls dedupe() with a key; find it.", S("def dedupe", "contacts")),
            ("Read dedupe().", V("contacts/iterutils.py", 1, 9)),
            ("It checks the computed key but records the raw item in `seen`, so keyed dedupe never matches and "
             "unhashable items crash. Record the key.",
             R("contacts/iterutils.py", "        seen.add(item)\n", "        seen.add(k)\n")),
            ("Re-run the repro.",
             B("python -c 'from contacts.importer import import_contacts; "
               "print(import_contacts([{\"email\": \"a@x.io\"}, {\"email\": \"A@x.io \"}, {\"email\": \"b@x.io\"}]))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 37 ---------------------------------------------------------------------------------------
    dict(
        id="seed-037-flatten-string-recursion",
        bug_type="recursion-base-case", difficulty="easy",
        issue="""\
            RecursionError when collecting tags

            `all_tags(posts)` dies as soon as any post has tags:

            ```
            >>> all_tags([{"tags": ["python", ["tips", "async"]]}])
              ...
              File ".../nested/flatten.py", line 10, in flatten
                out.extend(flatten(item))
              File ".../nested/flatten.py", line 6, in flatten
                if not isinstance(obj, Iterable):
            RecursionError: maximum recursion depth exceeded
            ```

            It works for lists of numbers.
            """,
        files={
            "nested/__init__.py": "",
            "nested/flatten.py": '''\
                from collections.abc import Iterable


                def flatten(obj):
                    """Recursively flatten nested iterables into a flat list; a scalar becomes [obj]."""
                    if not isinstance(obj, Iterable):
                        return [obj]
                    out = []
                    for item in obj:
                        out.extend(flatten(item))
                    return out
                ''',
            "nested/tags.py": '''\
                from .flatten import flatten


                def all_tags(posts):
                    """Sorted unique tags across posts; tags may be nested lists."""
                    return sorted(set(flatten([p.get("tags", []) for p in posts])))
                ''',
            "tests/test_flatten.py": '''\
                from nested.flatten import flatten


                def test_nested_numbers():
                    assert flatten([1, [2, [3, (4, 5)]], []]) == [1, 2, 3, 4, 5]


                def test_scalar():
                    assert flatten(7) == [7]


                def test_empty():
                    assert flatten([]) == []
                ''',
        },
        hidden_test='''\
            from nested.flatten import flatten
            from nested.tags import all_tags


            def test_strings_are_atoms():
                assert flatten(["ab", ["cd", ("e",)]]) == ["ab", "cd", "e"]
                assert flatten("hello") == ["hello"]
                assert flatten([b"raw"]) == [b"raw"]


            def test_all_tags():
                posts = [{"tags": ["python", ["tips", "async"]]}, {"tags": ["python"]}, {}]
                assert all_tags(posts) == ["async", "python", "tips"]


            def test_numbers_still_flatten():
                assert flatten([[1], [2, [3]]]) == [1, 2, 3]
            ''',
        p2p=["tests/test_flatten.py::test_nested_numbers", "tests/test_flatten.py::test_scalar",
             "tests/test_flatten.py::test_empty"],
        steps=[
            ("Find flatten.", F("flatten.py")),
            ("Read it.", V("nested/flatten.py")),
            ("Reproduce with a single string.",
             B("python -c 'from nested.flatten import flatten; print(flatten([\"ab\"]))' 2>&1 | tail -1")),
            ("A string is Iterable and each character is itself a 1-char string, so the recursion never bottoms "
             "out. Treat str/bytes as atoms.",
             R("nested/flatten.py", "    if not isinstance(obj, Iterable):\n",
               "    if isinstance(obj, (str, bytes)) or not isinstance(obj, Iterable):\n")),
            ("Verify with the issue's data through all_tags.",
             B("python -c 'from nested.tags import all_tags; print(all_tags([{\"tags\": [\"python\", [\"tips\", \"async\"]]}, {}]))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 38 ---------------------------------------------------------------------------------------
    dict(
        id="seed-038-edit-distance-row-aliasing",
        bug_type="list-aliasing", difficulty="hard",
        issue="""\
            "Did you mean" suggestions are mostly missing

            Our CLI's typo suggestions basically never fire. `closest("comit", ["commit",
            "config"])` returns None, and `closest("pul", ["pull", "push"])` does return "pull"
            but the reported distance is 2. Even `levenshtein("abc", "abc")` comes back as 2
            instead of 0. Not sure when this regressed - some trivial cases (empty strings,
            single characters) look right.
            """,
        files={
            "fuzzy/__init__.py": "",
            "fuzzy/distance.py": '''\
                def levenshtein(a, b):
                    """Minimum number of single-character insertions, deletions and substitutions."""
                    n, m = len(a), len(b)
                    dp = [[0] * (m + 1)] * (n + 1)
                    for i in range(n + 1):
                        dp[i][0] = i
                    for j in range(m + 1):
                        dp[0][j] = j
                    for i in range(1, n + 1):
                        for j in range(1, m + 1):
                            cost = 0 if a[i - 1] == b[j - 1] else 1
                            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
                    return dp[n][m]


                def similarity(a, b):
                    if not a and not b:
                        return 1.0
                    return 1 - levenshtein(a, b) / max(len(a), len(b))
                ''',
            "fuzzy/suggest.py": '''\
                from .distance import levenshtein


                def closest_with_distance(word, candidates, max_distance=2):
                    best = None
                    for c in candidates:
                        d = levenshtein(word.lower(), c.lower())
                        if d <= max_distance and (best is None or d < best[0]):
                            best = (d, c)
                    return best


                def closest(word, candidates, max_distance=2):
                    """Best candidate within max_distance edits, or None."""
                    best = closest_with_distance(word, candidates, max_distance)
                    return best[1] if best else None
                ''',
            "tests/test_fuzzy.py": '''\
                from fuzzy.distance import levenshtein, similarity
                from fuzzy.suggest import closest


                def test_empty_strings():
                    assert levenshtein("", "abc") == 3
                    assert similarity("", "") == 1.0


                def test_single_char():
                    assert levenshtein("a", "b") == 1


                def test_no_candidate_within_range():
                    assert closest("zzzzzz", ["pull", "push"]) is None
                ''',
        },
        hidden_test='''\
            from fuzzy.distance import levenshtein, similarity
            from fuzzy.suggest import closest, closest_with_distance


            def test_classic_distances():
                assert levenshtein("kitten", "sitting") == 3
                assert levenshtein("flaw", "lawn") == 2
                assert levenshtein("abc", "abc") == 0
                assert levenshtein("abc", "") == 3


            def test_symmetric():
                for a, b in [("stats", "status"), ("comit", "remit"), ("ab", "ba")]:
                    assert levenshtein(a, b) == levenshtein(b, a)


            def test_suggestions():
                assert closest("comit", ["commit", "config"]) == "commit"
                assert closest_with_distance("pul", ["pull", "push"]) == (1, "pull")
                assert similarity("abcd", "abcd") == 1.0


            def test_far_word_still_rejected():
                assert closest("zzzzzz", ["commit"]) is None
            ''',
        p2p=["tests/test_fuzzy.py::test_empty_strings", "tests/test_fuzzy.py::test_single_char",
             "tests/test_fuzzy.py::test_no_candidate_within_range"],
        steps=[
            ("Start at the suggestion code.", V("fuzzy/suggest.py")),
            ("Reproduce the bad suggestions.",
             B("python -c 'from fuzzy.suggest import closest, closest_with_distance as cw; "
               "print(closest(\"comit\", [\"commit\", \"config\"]), cw(\"pul\", [\"pull\", \"push\"]))'")),
            ("Selection logic is straightforward; the distances themselves look wrong. Check levenshtein directly.",
             B("python -c 'from fuzzy.distance import levenshtein as L; print(L(\"abc\",\"abc\"), L(\"abc\",\"\"), "
               "L(\"kitten\",\"sitting\"), L(\"\",\"abc\"))'")),
            ("L('abc', '') == 0 means the first-column initialization is being clobbered. Read the DP setup.",
             V("fuzzy/distance.py", 1, 13)),
            ("`[[0] * (m + 1)] * (n + 1)` repeats the same row object n+1 times. Confirm the aliasing.",
             B("python -c 'dp = [[0] * 3] * 2; dp[1][0] = 9; print(dp, dp[0] is dp[1])'")),
            ("Every row is the same list, so writes to dp[0][j] overwrite the column inits and each row reads "
             "itself as the previous row. Allocate an independent list per row.",
             R("fuzzy/distance.py", "    dp = [[0] * (m + 1)] * (n + 1)\n",
               "    dp = [[0] * (m + 1) for _ in range(n + 1)]\n")),
            ("Re-check the distances.",
             B("python -c 'from fuzzy.distance import levenshtein as L; print(L(\"abc\",\"abc\"), L(\"abc\",\"\"), "
               "L(\"kitten\",\"sitting\"), L(\"flaw\",\"lawn\"))'")),
            ("And the suggestions from the issue.",
             B("python -c 'from fuzzy.suggest import closest, closest_with_distance as cw; "
               "print(closest(\"comit\", [\"commit\", \"config\"]), cw(\"pul\", [\"pull\", \"push\"]))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("DP table rows are no longer aliased.", SUBMIT),
        ],
    ),
    # 39 ---------------------------------------------------------------------------------------
    dict(
        id="seed-039-toposort-duplicate-edge-indegree",
        bug_type="graph-invariant", difficulty="hard",
        issue="""\
            install_order reports a dependency cycle that doesn't exist

            Since we started merging `requires` and `extras` into one manifest, some packages list
            the same dependency twice, and install_order blows up even though there's no cycle:

            ```
            >>> install_order({"web": ["json", "http", "json"], "http": ["json"], "json": []})
            Traceback (most recent call last):
              ...
              File ".../depgraph/toposort.py", line 22, in topo_order
                raise CycleError(f"dependency cycle among: {', '.join(stuck)}")
            depgraph.toposort.CycleError: dependency cycle among: web
            ```

            Expected `['json', 'http', 'web']`. Removing the duplicate "json" makes it work.
            """,
        files={
            "depgraph/__init__.py": "",
            "depgraph/graph.py": '''\
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
                ''',
            "depgraph/toposort.py": '''\
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
                ''',
            "depgraph/manifest.py": '''\
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
                ''',
            "tests/test_depgraph.py": '''\
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
                ''',
        },
        hidden_test='''\
            import pytest

            from depgraph.graph import DepGraph
            from depgraph.manifest import install_order
            from depgraph.toposort import CycleError, topo_order


            def test_duplicate_dependency_is_not_a_cycle():
                m = {"web": ["json", "http", "json"], "http": ["json"], "json": []}
                assert install_order(m) == ["json", "http", "web"]


            def test_repeated_add_dependency_counts_once():
                g = DepGraph()
                g.add_dependency("app", "lib")
                g.add_dependency("app", "lib")
                assert g.indegree["app"] == 1
                assert topo_order(g) == ["lib", "app"]


            def test_cycle_still_detected_with_duplicates():
                with pytest.raises(CycleError):
                    install_order({"a": ["b", "b"], "b": ["c"], "c": ["a"]})
            ''',
        p2p=["tests/test_depgraph.py::test_simple_chain", "tests/test_depgraph.py::test_diamond",
             "tests/test_depgraph.py::test_real_cycle", "tests/test_depgraph.py::test_dependents"],
        steps=[
            ("Reproduce with the manifest from the issue.",
             B("python -c 'from depgraph.manifest import install_order; "
               "print(install_order({\"web\": [\"json\", \"http\", \"json\"], \"http\": [\"json\"], \"json\": []}))' 2>&1 | tail -1")),
            ("The error comes from Kahn's algorithm. Read it.", V("depgraph/toposort.py")),
            ("topo_order decrements once per successor, and successors come from the graph. Look at how edges "
             "and in-degrees are recorded.", V("depgraph/graph.py")),
            ("Confirm the counts disagree for the duplicated edge.",
             B("python -c 'from depgraph.manifest import build_graph; "
               "g=build_graph({\"web\": [\"json\", \"http\", \"json\"], \"http\": [\"json\"], \"json\": []}); "
               "print(dict(g.indegree), {k: sorted(v) for k, v in g.successors.items()})'")),
            ("`successors` is a set, so the duplicate edge is stored once, but indegree is bumped twice; web can "
             "never reach 0. Check nothing else touches indegree before fixing it in the graph.",
             S("indegree", "depgraph")),
            ("Only count an edge the first time it is added, keeping indegree consistent with successors.",
             R("depgraph/graph.py",
               "        self.successors[dep].add(pkg)\n        self.indegree[pkg] += 1\n",
               "        if pkg not in self.successors[dep]:\n            self.successors[dep].add(pkg)\n"
               "            self.indegree[pkg] += 1\n")),
            ("Re-run the repro.",
             B("python -c 'from depgraph.manifest import install_order; "
               "print(install_order({\"web\": [\"json\", \"http\", \"json\"], \"http\": [\"json\"], \"json\": []}))'")),
            ("Make sure a real cycle with duplicates is still detected.",
             B("python -c 'from depgraph.manifest import install_order; install_order({\"a\": [\"b\", \"b\"], \"b\": [\"a\"]})' 2>&1 | tail -1")),
            ("Run the tests.", T()),
            ("Fixed in the graph so every consumer sees consistent in-degrees.", SUBMIT),
        ],
    ),
    # 40 ---------------------------------------------------------------------------------------
    dict(
        id="seed-040-indexed-heap-stale-position",
        bug_type="index-bookkeeping", difficulty="hard",
        issue="""\
            Scheduler runs tasks out of deadline order after reschedule()

            We see tasks occasionally executing in the wrong order once deadlines get moved, and
            once in a while `reschedule` throws `IndexError: list index out of range`. Smallest
            repro I could get:

                s = Scheduler()
                s.submit("backup", 10)
                s.submit("report", 20)
                s.next_task()              # 'backup'
                s.submit("email", 30)
                s.reschedule("report", 40)
                s.run_all()                # ['report', 'email'], expected ['email', 'report']

            Without the `next_task()` call in the middle the order is correct.
            """,
        files={
            "taskq/__init__.py": "",
            "taskq/heap.py": '''\
                class IndexedHeap:
                    """Binary min-heap of (priority, key) with O(log n) priority updates by key."""

                    def __init__(self):
                        self._heap = []  # list of [priority, key]
                        self._pos = {}   # key -> index in self._heap

                    def __len__(self):
                        return len(self._heap)

                    def __contains__(self, key):
                        return key in self._pos

                    def push(self, key, priority):
                        if key in self._pos:
                            raise KeyError(f"{key!r} is already queued")
                        self._heap.append([priority, key])
                        self._pos[key] = len(self._heap) - 1
                        self._sift_up(len(self._heap) - 1)

                    def pop(self):
                        """Remove and return (key, priority) with the smallest priority."""
                        if not self._heap:
                            raise IndexError("pop from empty heap")
                        top = self._heap[0]
                        last = self._heap.pop()
                        del self._pos[top[1]]
                        if self._heap:
                            self._heap[0] = last
                            self._sift_down(0)
                        return top[1], top[0]

                    def update(self, key, priority):
                        i = self._pos[key]
                        old = self._heap[i][0]
                        self._heap[i][0] = priority
                        if priority < old:
                            self._sift_up(i)
                        else:
                            self._sift_down(i)

                    def _swap(self, i, j):
                        h = self._heap
                        h[i], h[j] = h[j], h[i]
                        self._pos[h[i][1]] = i
                        self._pos[h[j][1]] = j

                    def _sift_up(self, i):
                        while i > 0:
                            parent = (i - 1) // 2
                            if self._heap[i][0] < self._heap[parent][0]:
                                self._swap(i, parent)
                                i = parent
                            else:
                                break

                    def _sift_down(self, i):
                        n = len(self._heap)
                        while True:
                            smallest = i
                            for child in (2 * i + 1, 2 * i + 2):
                                if child < n and self._heap[child][0] < self._heap[smallest][0]:
                                    smallest = child
                            if smallest == i:
                                return
                            self._swap(i, smallest)
                            i = smallest
                ''',
            "taskq/scheduler.py": '''\
                from .heap import IndexedHeap


                class Scheduler:
                    """Runs tasks in deadline order; deadlines can be moved while a task is queued."""

                    def __init__(self):
                        self._queue = IndexedHeap()

                    def submit(self, task, deadline):
                        self._queue.push(task, deadline)

                    def reschedule(self, task, deadline):
                        self._queue.update(task, deadline)

                    def next_task(self):
                        return self._queue.pop()[0]

                    def pending(self):
                        return len(self._queue)

                    def run_all(self):
                        out = []
                        while self.pending():
                            out.append(self.next_task())
                        return out
                ''',
            "tests/test_scheduler.py": '''\
                import pytest

                from taskq.heap import IndexedHeap
                from taskq.scheduler import Scheduler


                def test_runs_in_deadline_order():
                    s = Scheduler()
                    for task, deadline in [("c", 30), ("a", 10), ("d", 40), ("b", 20)]:
                        s.submit(task, deadline)
                    assert s.run_all() == ["a", "b", "c", "d"]


                def test_reschedule_before_any_pop():
                    s = Scheduler()
                    s.submit("a", 10)
                    s.submit("b", 20)
                    s.submit("c", 30)
                    s.reschedule("c", 5)
                    s.reschedule("a", 25)
                    assert s.run_all() == ["c", "b", "a"]


                def test_duplicate_push_rejected():
                    h = IndexedHeap()
                    h.push("x", 1)
                    with pytest.raises(KeyError):
                        h.push("x", 2)


                def test_pop_empty():
                    with pytest.raises(IndexError):
                        IndexedHeap().pop()
                ''',
        },
        hidden_test='''\
            from taskq.heap import IndexedHeap
            from taskq.scheduler import Scheduler


            def _check_positions(h):
                for key, i in h._pos.items():
                    assert h._heap[i][1] == key


            def test_reschedule_after_pop():
                s = Scheduler()
                s.submit("backup", 10)
                s.submit("report", 20)
                assert s.next_task() == "backup"
                s.submit("email", 30)
                s.reschedule("report", 40)
                assert s.run_all() == ["email", "report"]


            def test_positions_consistent_through_mixed_ops():
                h = IndexedHeap()
                for i, p in enumerate([5, 3, 8, 1, 9, 2, 7]):
                    h.push(f"k{i}", p)
                for step in range(5):
                    h.pop()
                    _check_positions(h)
                    if len(h):
                        key = next(iter(h._pos))
                        h.update(key, 100 - step)
                        _check_positions(h)
                    h.push(f"n{step}", step * 3)
                    _check_positions(h)


            def test_heap_sort_order():
                h = IndexedHeap()
                for i, p in enumerate([4, 1, 3, 2]):
                    h.push(i, p)
                assert [h.pop()[1] for _ in range(4)] == [1, 2, 3, 4]
            ''',
        p2p=["tests/test_scheduler.py::test_runs_in_deadline_order", "tests/test_scheduler.py::test_reschedule_before_any_pop",
             "tests/test_scheduler.py::test_duplicate_push_rejected", "tests/test_scheduler.py::test_pop_empty"],
        steps=[
            ("The scheduler is a thin wrapper; confirm what it delegates to.", V("taskq/scheduler.py")),
            ("Reproduce the issue's sequence.",
             B("python -c 'from taskq.scheduler import Scheduler; s=Scheduler(); s.submit(\"backup\",10); "
               "s.submit(\"report\",20); s.next_task(); s.submit(\"email\",30); s.reschedule(\"report\",40); print(s.run_all())'")),
            ("Only wrong after a pop, so suspect the key->index map going stale. Read the heap.",
             V("taskq/heap.py", 14, 39)),
            ("Dump the internal state right after the pop.",
             B("python -c 'from taskq.heap import IndexedHeap; h=IndexedHeap(); h.push(\"backup\",10); h.push(\"report\",20); "
               "h.pop(); print(h._heap, h._pos)'")),
            ("After pop, `report` sits at index 0 but _pos still says 1. pop() moves the last element to the root "
             "without updating _pos; _swap fixes it only if sift-down actually moves it. Record the new position.",
             I("taskq/heap.py", 29, "            self._pos[last[1]] = 0")),
            ("Check the edit landed in the right place.", V("taskq/heap.py", 21, 32)),
            ("Re-run the repro.",
             B("python -c 'from taskq.scheduler import Scheduler; s=Scheduler(); s.submit(\"backup\",10); "
               "s.submit(\"report\",20); s.next_task(); s.submit(\"email\",30); s.reschedule(\"report\",40); print(s.run_all())'")),
            ("Run the tests.", T()),
            ("Position map is now consistent after pop.", SUBMIT),
        ],
    ),
]
