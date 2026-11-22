from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LineItem:
    sku: str
    unit_cents: int
    qty: int = 1

    @property
    def total_cents(self):
        return self.unit_cents * self.qty


@dataclass
class Order:
    lines: list = field(default_factory=list)
    region: str = "CA"
    coupon_code: Optional[str] = None
    discount_cents: int = 0
    shipping_cents: int = 0
    tax_cents: int = 0

    @property
    def subtotal_cents(self):
        return sum(line.total_cents for line in self.lines)

    @property
    def discounted_subtotal_cents(self):
        """What the customer pays for goods, after order-level discounts."""
        return self.subtotal_cents - self.discount_cents

    @property
    def total_cents(self):
        return self.discounted_subtotal_cents + self.shipping_cents + self.tax_cents
