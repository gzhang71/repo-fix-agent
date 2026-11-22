def c_to_k(c):
    return c + 273.15


def k_to_c(k):
    return k - 273.15


def f_to_k(f):
    return (f - 32) * 5 / 9 + 273.15


def k_to_f(k):
    return (k - 273.15) * 9 / 5 - 32


_TO_K = {"C": c_to_k, "F": f_to_k, "K": lambda k: k}
_FROM_K = {"C": k_to_c, "F": k_to_f, "K": lambda k: k}


def convert_temp(value, src, dst):
    if src not in _TO_K or dst not in _FROM_K:
        raise ValueError(f"unsupported unit: {src} -> {dst}")
    return _FROM_K[dst](_TO_K[src](value))
