from .csvline import split_line


def load_contacts(text):
    """Parse CSV text with a header row into a list of dicts."""
    lines = [l.rstrip("\r") for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    header = [h.strip().lower() for h in split_line(lines[0])]
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        values = split_line(line)
        if len(values) != len(header):
            raise ValueError(f"line {n}: expected {len(header)} fields, got {len(values)}")
        rows.append(dict(zip(header, values)))
    return rows
