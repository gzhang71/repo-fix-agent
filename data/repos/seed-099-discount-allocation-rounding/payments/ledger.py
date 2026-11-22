class RefundError(Exception):
    pass


class Ledger:
    def __init__(self):
        self.captured = {}
        self.refunded = {}

    def capture(self, order_id, cents):
        self.captured[order_id] = self.captured.get(order_id, 0) + cents

    def remaining(self, order_id):
        return self.captured.get(order_id, 0) - self.refunded.get(order_id, 0)

    def refund(self, order_id, cents):
        if cents > self.remaining(order_id):
            raise RefundError(
                f"refund of {cents} exceeds remaining captured amount "
                f"{self.remaining(order_id)} for order {order_id}")
        self.refunded[order_id] = self.refunded.get(order_id, 0) + cents
