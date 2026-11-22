from decimal import ROUND_HALF_UP, Decimal

from .clock import shift_minutes


def hours_worked(shifts):
    return Decimal(sum(shift_minutes(start, end) for start, end in shifts)) / 60


def gross_pay(shifts, hourly_rate):
    pay = hours_worked(shifts) * Decimal(str(hourly_rate))
    return pay.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
