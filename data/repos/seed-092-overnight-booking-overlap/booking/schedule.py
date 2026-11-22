def day_schedule(repo, room, day):
    """Rows for the day view: (booking id, start, end) as HH:MM strings."""
    return [(b.id, b.start.strftime("%H:%M"), b.end.strftime("%H:%M"))
            for b in repo.for_room_on(room, day)]
