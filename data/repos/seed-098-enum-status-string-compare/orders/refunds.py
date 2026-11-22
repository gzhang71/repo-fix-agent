RESTOCKING_FEE_PCT = 15


def refund_amount(order):
    """Full refund for orders that never left the warehouse; returned
    (delivered) orders are refunded minus the restocking fee."""
    if order.status == "delivered":
        return order.total_cents - order.total_cents * RESTOCKING_FEE_PCT // 100
    return order.total_cents
