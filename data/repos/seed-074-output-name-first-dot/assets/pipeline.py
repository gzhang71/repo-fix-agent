from .naming import output_name


def plan_conversions(sources, new_ext):
    """Map each source path to its converted output path."""
    plan = {}
    seen = set()
    for src in sources:
        dst = output_name(src, new_ext)
        if dst in seen:
            raise ValueError(f"two inputs map to the same output {dst!r}")
        seen.add(dst)
        plan[src] = dst
    return plan
