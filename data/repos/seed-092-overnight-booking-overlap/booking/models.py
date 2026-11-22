from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Booking:
    id: int
    room: str
    start: datetime
    end: datetime

    def overlaps(self, other):
        """Half-open intervals: back-to-back bookings do not overlap."""
        return self.start < other.end and other.start < self.end
