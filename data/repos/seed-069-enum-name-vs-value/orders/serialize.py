from .order import Order
from .status import OrderStatus


def to_dict(order):
    return {"id": order.id, "status": order.status.name}


def from_dict(data):
    return Order(id=int(data["id"]), status=OrderStatus(data["status"]))
