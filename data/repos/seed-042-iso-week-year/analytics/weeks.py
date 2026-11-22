from datetime import timedelta


def week_key(d):
    """ISO-8601 week label such as '2024-W07' used to bucket events."""
    return f"{d.year}-W{d.isocalendar()[1]:02d}"


def week_start(d):
    """Monday of the week containing d."""
    return d - timedelta(days=d.weekday())
