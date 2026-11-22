from collections.abc import Iterable


def flatten(obj):
    """Recursively flatten nested iterables into a flat list; a scalar becomes [obj]."""
    if not isinstance(obj, Iterable):
        return [obj]
    out = []
    for item in obj:
        out.extend(flatten(item))
    return out
