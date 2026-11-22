def counted(fn):
    """Count how many times a command has been invoked (exposed as `.calls`)."""
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
