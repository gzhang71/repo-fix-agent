def apply_change(a, change):
    """Apply a Change from diff_lines(a, b) to `a`, returning b."""
    if change is None:
        return list(a)
    end = change.start_a + len(change.removed)
    if a[change.start_a:end] != change.removed:
        raise ValueError("change does not apply: context mismatch")
    return a[:change.start_a] + change.added + a[end:]
