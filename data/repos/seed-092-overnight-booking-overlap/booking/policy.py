from datetime import timedelta

MAX_DURATION = timedelta(hours=12)


class BookingError(ValueError):
    pass


def validate(start, end):
    if end <= start:
        raise BookingError("booking must end after it starts")
    if end - start > MAX_DURATION:
        raise BookingError("bookings are limited to 12 hours")
