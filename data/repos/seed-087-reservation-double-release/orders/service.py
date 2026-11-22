from .models import Order

HOLD_TICKS = 30


class OrderService:
    def __init__(self, book):
        self.book = book
        self.orders = {}

    def place(self, order_id, items, now):
        self.book.reserve(order_id, items, expires_at=now + HOLD_TICKS)
        order = Order(order_id, dict(items))
        self.orders[order_id] = order
        return order

    def cancel(self, order_id):
        order = self.orders[order_id]
        if order.status != "pending":
            raise ValueError(f"cannot cancel a {order.status} order")
        self.book.release(order_id)
        order.status = "cancelled"
        return order

    def ship(self, order_id):
        order = self.orders[order_id]
        self.book.commit(order_id)
        order.status = "shipped"
        return order

    def sweep(self, now):
        for order_id in self.book.expire(now):
            self.orders[order_id].status = "expired"
