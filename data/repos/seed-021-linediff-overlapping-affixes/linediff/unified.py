from .core import diff_lines


def unified(a, b, name_a="a", name_b="b"):
    change = diff_lines(a, b)
    if change is None:
        return ""
    out = [f"--- {name_a}", f"+++ {name_b}",
           f"@@ -{change.start_a + 1},{len(change.removed)} +{change.start_b + 1},{len(change.added)} @@"]
    out += ["-" + line for line in change.removed]
    out += ["+" + line for line in change.added]
    return "\n".join(out) + "\n"


def summary(a, b):
    change = diff_lines(a, b)
    if change is None:
        return "no changes"
    return f"{len(change.added)} added, {len(change.removed)} removed"
