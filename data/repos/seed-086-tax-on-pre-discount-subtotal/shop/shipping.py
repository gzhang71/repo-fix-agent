FREE_SHIPPING_CENTS = 5000
FLAT_RATE_CENTS = 599


def shipping_for(order):
    if order.discounted_subtotal_cents >= FREE_SHIPPING_CENTS:
        return 0
    return FLAT_RATE_CENTS
