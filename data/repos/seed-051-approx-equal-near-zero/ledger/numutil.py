def approx_equal(a, b, rel_tol=1e-9):
    """True if a and b are equal up to floating-point noise."""
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))
