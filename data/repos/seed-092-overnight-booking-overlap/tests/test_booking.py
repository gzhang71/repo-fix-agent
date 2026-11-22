from datetime import date, datetime

import pytest

from booking.policy import BookingError
from booking.repository import BookingRepository
from booking.schedule import day_schedule
from booking.service import BookingService, Conflict


def dt(d, h, m=0):
    return datetime(2024, 5, d, h, m)


def test_same_day_conflict():
    svc = BookingService(BookingRepository())
    svc.book("A", dt(10, 9), dt(10, 11))
    with pytest.raises(Conflict):
        svc.book("A", dt(10, 10), dt(10, 12))


def test_back_to_back_allowed():
    svc = BookingService(BookingRepository())
    svc.book("A", dt(10, 10), dt(10, 11))
    svc.book("A", dt(10, 11), dt(10, 12))


def test_other_room_free():
    svc = BookingService(BookingRepository())
    svc.book("A", dt(10, 9), dt(10, 11))
    svc.book("B", dt(10, 9), dt(10, 11))


def test_policy():
    svc = BookingService(BookingRepository())
    with pytest.raises(BookingError):
        svc.book("A", dt(10, 9), dt(10, 23))


def test_day_schedule():
    repo = BookingRepository()
    BookingService(repo).book("A", dt(10, 14), dt(10, 15, 30))
    assert day_schedule(repo, "A", date(2024, 5, 10)) == [(1, "14:00", "15:30")]
