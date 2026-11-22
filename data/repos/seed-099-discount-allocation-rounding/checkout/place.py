from orders.models import Order
from pricing.allocation import allocate


def place_order(order_id, lines, ledger, discount_cents=0):
    order = Order(order_id, list(lines), discount_cents)
    if discount_cents > order.subtotal_cents:
        raise ValueError("discount exceeds order subtotal")
    order.allocations = allocate(discount_cents, [l.total_cents for l in order.lines])
    ledger.capture(order.id, order.paid_cents)
    return order
