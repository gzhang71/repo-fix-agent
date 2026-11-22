from .refunds import refund_amount
from .status import OrderStatus


class CannotCancel(Exception):
    pass


class CannotReturn(Exception):
    pass


class OrderService:
    def __init__(self, stock, gateway):
        self.stock = stock
        self.gateway = gateway
        self.orders = {}

    def create(self, order):
        self.stock.take(order.items)
        self.orders[order.id] = order
        return order

    def pay(self, order_id):
        order = self.orders[order_id]
        self.gateway.charge(order.id, order.total_cents)
        order.status = OrderStatus.PAID

    def ship(self, order_id):
        self.orders[order_id].status = OrderStatus.SHIPPED

    def deliver(self, order_id):
        self.orders[order_id].status = OrderStatus.DELIVERED

    def cancel(self, order_id):
        order = self.orders[order_id]
        if order.status in ("shipped", "delivered"):
            raise CannotCancel(f"order {order_id} has already shipped")
        if order.status == OrderStatus.CANCELLED:
            return order
        if order.status == OrderStatus.PAID:
            self.gateway.refund(order.id, refund_amount(order))
        self.stock.restock(order.items)
        order.status = OrderStatus.CANCELLED
        return order

    def return_order(self, order_id):
        order = self.orders[order_id]
        if order.status != OrderStatus.DELIVERED:
            raise CannotReturn(f"order {order_id} was not delivered")
        self.gateway.refund(order.id, refund_amount(order))
        self.stock.restock(order.items)
        order.status = OrderStatus.RETURNED
        return order
