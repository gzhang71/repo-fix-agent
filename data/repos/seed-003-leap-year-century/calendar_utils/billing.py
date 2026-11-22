from .dates import days_in_month


def prorate(amount, year, month, days_used):
    return round(amount * days_used / days_in_month(year, month), 2)
