from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

    @property
    def is_final(self):
        return self in (OrderStatus.SHIPPED, OrderStatus.CANCELLED)
