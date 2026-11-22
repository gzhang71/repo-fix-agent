from datetime import date


def add_months(d, months):
    """Return the same day `months` later, clamped to the end of the target month."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    return d.replace(year=year, month=month)


def month_start(d):
    return date(d.year, d.month, 1)
