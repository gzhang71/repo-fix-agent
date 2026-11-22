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
