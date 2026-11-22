from dataclasses import dataclass


class OutOfStock(Exception):
    pass


@dataclass
class Reservation:
    order_id: str
    items: dict
    expires_at: int


class ReservationBook:
    def __init__(self, stock):
        self.stock = stock
        self._by_order = {}

    def reserve(self, order_id, items, expires_at):
        for sku, qty in items.items():
            if self.stock.available(sku) < qty:
                raise OutOfStock(sku)
        for sku, qty in items.items():
            self.stock.reserved[sku] = self.stock.reserved.get(sku, 0) + qty
        self._by_order[order_id] = Reservation(order_id, dict(items), expires_at)

    def release(self, order_id):
        """Give reserved units back to the pool (cancellation or timeout)."""
        res = self._by_order.get(order_id)
        if res is None:
            return False
        for sku, qty in res.items.items():
            self.stock.reserved[sku] -= qty
        return True

    def commit(self, order_id):
        """Turn a reservation into a real stock decrement when the order ships."""
        res = self._by_order.pop(order_id)
        for sku, qty in res.items.items():
            self.stock.reserved[sku] -= qty
            self.stock.on_hand[sku] -= qty

    def expire(self, now):
        """Release every reservation whose hold has run out; return their order ids."""
        expired = [oid for oid, r in self._by_order.items() if r.expires_at <= now]
        for oid in expired:
            self.release(oid)
        return expired
