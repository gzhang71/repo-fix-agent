from datetime import datetime

from .models import Booking


class BookingRepository:
    def __init__(self):
        self._rows = []
        self._next_id = 1

    def add(self, room, start, end):
        booking = Booking(self._next_id, room, start, end)
        self._next_id += 1
        self._rows.append(booking)
        return booking

    def for_room(self, room):
        return sorted((b for b in self._rows if b.room == room), key=lambda b: b.start)

    def for_room_on(self, room, day):
        """Bookings for `room` that touch calendar day `day` (a date), by start time."""
        return [b for b in self.for_room(room) if b.start.date() == day]
