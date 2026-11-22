from dataclasses import dataclass


@dataclass
class Change:
    """Replace a[start_a:start_a + len(removed)] with `added` (inserted at b[start_b])."""
    start_a: int
    removed: list
    start_b: int
    added: list


def _common_affixes(a, b):
    prefix = 0
    while prefix < len(a) and prefix < len(b) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(a) and suffix < len(b) and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    return prefix, suffix


def diff_lines(a, b):
    """Single change spanning everything between the common prefix and suffix, or None."""
    prefix, suffix = _common_affixes(a, b)
    removed = a[prefix:len(a) - suffix]
    added = b[prefix:len(b) - suffix]
    if not removed and not added:
        return None
    return Change(prefix, removed, prefix, added)
