from dataclasses import dataclass, field

from .status import OrderStatus


@dataclass
class Order:
    id: int
    items: dict = field(default_factory=dict)
    total_cents: int = 0
    status: OrderStatus = OrderStatus.PENDING
