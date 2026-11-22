HEADER = ["id", "name", "email", "notes"]


def rows_to_csv(rows, out):
    """Write contact dicts to the text stream `out` as CSV with a header row."""
    out.write(",".join(HEADER) + "\n")
    for row in rows:
        out.write(",".join(str(row.get(col, "")) for col in HEADER) + "\n")
