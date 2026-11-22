import operator

_BINARY = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def evaluate(node):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "neg":
        return -evaluate(node[1])
    left, right = evaluate(node[1]), evaluate(node[2])
    if kind == "/" and right == 0:
        raise ZeroDivisionError("division by zero in expression")
    return _BINARY[kind](left, right)
