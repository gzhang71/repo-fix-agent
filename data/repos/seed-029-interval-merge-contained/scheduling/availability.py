from .intervals import merge


def free_slots(busy, day_start, day_end, min_length=0):
    """Return the gaps between busy intervals within [day_start, day_end]."""
    slots = []
    cursor = day_start
    for start, end in merge(busy):
        if end <= day_start or start >= day_end:
            continue
        if start > cursor and start - cursor >= min_length:
            slots.append((cursor, start))
        cursor = max(cursor, end)
    if day_end > cursor and day_end - cursor >= min_length:
        slots.append((cursor, day_end))
    return slots
