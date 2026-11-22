class AlreadyRefunded(Exception):
    pass


def refundable_amount(order, index):
    """What the customer actually paid for a line: its price minus its discount share."""
    return order.lines[index].total_cents - order.allocations[index]


def refund_line(order, index, ledger):
    line = order.lines[index]
    if line.refunded:
        raise AlreadyRefunded(f"line {index} of order {order.id}")
    amount = refundable_amount(order, index)
    ledger.refund(order.id, amount)
    line.refunded = True
    return amount
