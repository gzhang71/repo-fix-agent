from .flatten import flatten


def all_tags(posts):
    """Sorted unique tags across posts; tags may be nested lists."""
    return sorted(set(flatten([p.get("tags", []) for p in posts])))
