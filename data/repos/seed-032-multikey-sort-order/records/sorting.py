def sort_by(rows, *fields):
    """Sort dict rows by several fields. Prefix a field with '-' for descending.

    sort_by(rows, "dept", "-salary") sorts by dept ascending, then by salary
    descending within each dept. The sort is stable.
    """
    result = list(rows)
    for spec in fields:
        name = spec.lstrip("-")
        result.sort(key=lambda r: r[name], reverse=spec.startswith("-"))
    return result
