from dataclasses import dataclass, field


@dataclass
class Line:
    sku: str
    unit_cents: int
    qty: int = 1
    refunded: bool = False

    @property
    def total_cents(self):
        return self.unit_cents * self.qty


@dataclass
class Order:
    id: int
    lines: list
    discount_cents: int = 0
    allocations: list = field(default_factory=list)  # discount share per line

    @property
    def subtotal_cents(self):
        return sum(line.total_cents for line in self.lines)

    @property
    def paid_cents(self):
        return self.subtotal_cents - self.discount_cents
