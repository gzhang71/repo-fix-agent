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
