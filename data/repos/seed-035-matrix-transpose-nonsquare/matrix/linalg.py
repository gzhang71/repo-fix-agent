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
