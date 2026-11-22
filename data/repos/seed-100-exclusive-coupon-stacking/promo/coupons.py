from dataclasses import dataclass


@dataclass(frozen=True)
class Coupon:
    code: str
    percent: int = 0
    amount_cents: int = 0
    exclusive: bool = False


CATALOG = {c.code: c for c in [
    Coupon("WELCOME5", amount_cents=500),
    Coupon("SPRING10", percent=10),
    Coupon("VIP20", percent=20, exclusive=True),
    Coupon("STAFF30", percent=30, exclusive=True),
]}


class InvalidCoupon(KeyError):
    pass


def lookup(code):
    try:
        return CATALOG[code.strip().upper()]
    except KeyError:
        raise InvalidCoupon(code) from None
