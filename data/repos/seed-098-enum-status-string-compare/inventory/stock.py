class Stock:
    def __init__(self, levels):
        self.levels = dict(levels)

    def take(self, items):
        for sku, qty in items.items():
            if self.levels.get(sku, 0) < qty:
                raise ValueError(f"not enough {sku}")
        for sku, qty in items.items():
            self.levels[sku] -= qty

    def restock(self, items):
        for sku, qty in items.items():
            self.levels[sku] = self.levels.get(sku, 0) + qty
