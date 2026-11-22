from .models import Booking
from .policy import validate


class Conflict(Exception):
    pass


class BookingService:
    def __init__(self, repo):
        self.repo = repo

    def book(self, room, start, end):
        validate(start, end)
        candidate = Booking(0, room, start, end)
        for existing in self.repo.for_room_on(room, start.date()):
            if existing.overlaps(candidate):
                raise Conflict(f"{room} is taken by booking {existing.id}")
        return self.repo.add(room, start, end)
