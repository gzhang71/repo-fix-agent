from dataclasses import dataclass

from .status import OrderStatus


@dataclass
class Order:
    id: int
    status: OrderStatus = OrderStatus.PENDING

    def advance(self, new_status):
        if self.status.is_final:
            raise ValueError(f"order {self.id} is already {self.status.value}")
        self.status = OrderStatus(new_status)
