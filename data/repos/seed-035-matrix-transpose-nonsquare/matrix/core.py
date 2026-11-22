def shape(m):
    return (len(m), len(m[0]) if m else 0)


def transpose(m):
    n = len(m)
    return [[m[i][j] for i in range(n)] for j in range(n)]


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
