from promo.coupons import lookup


class Cart:
    def __init__(self):
        self.items = []
        self.codes = []

    def add(self, sku, unit_cents, qty=1):
        self.items.append((sku, unit_cents, qty))

    def add_code(self, code):
        lookup(code)  # validate early so the UI can show an error
        self.codes.append(code)

    @property
    def subtotal_cents(self):
        return sum(unit * qty for _, unit, qty in self.items)
