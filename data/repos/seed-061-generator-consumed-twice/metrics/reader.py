def read_latencies(lines):
    """Parse 'endpoint,ms' log lines lazily, skipping blanks and comments."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        endpoint, ms = line.split(",")
        yield endpoint.strip(), float(ms)
