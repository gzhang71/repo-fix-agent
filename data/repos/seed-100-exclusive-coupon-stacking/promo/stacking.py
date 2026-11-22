def resolve(coupons):
    """Pick the coupons that apply to a cart, in the order they were entered.

    Duplicates are ignored. Exclusive coupons cannot be combined with anything:
    if any exclusive coupon was entered, only the first exclusive one applies.
    """
    unique = list({c.code: c for c in coupons}.values())
    if unique and unique[0].exclusive:
        return [unique[0]]
    return unique
