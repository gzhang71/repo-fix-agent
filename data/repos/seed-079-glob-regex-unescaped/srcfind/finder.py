from .patterns import glob_to_regex


def select(paths, include, exclude=()):
    """Paths matching any include glob and no exclude glob, sorted."""
    inc = [glob_to_regex(p) for p in include]
    exc = [glob_to_regex(p) for p in exclude]
    return sorted(p for p in paths
                  if any(r.match(p) for r in inc) and not any(r.match(p) for r in exc))
