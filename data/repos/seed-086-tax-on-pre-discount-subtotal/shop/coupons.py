COUPONS = {
    "SAVE10": {"kind": "percent", "value": 10, "min_subtotal_cents": 0},
    "FIVEOFF": {"kind": "fixed", "value": 500, "min_subtotal_cents": 2500},
}


class CouponError(ValueError):
    pass


def discount_for(code, subtotal_cents):
    coupon = COUPONS.get(code.upper())
    if coupon is None:
        raise CouponError(f"unknown coupon {code!r}")
    if subtotal_cents < coupon["min_subtotal_cents"]:
        raise CouponError(f"{code} requires a larger order")
    if coupon["kind"] == "percent":
        return subtotal_cents * coupon["value"] // 100
    return min(coupon["value"], subtotal_cents)
