class Stock:
    def __init__(self, on_hand=None):
        self.on_hand = dict(on_hand or {})
        self.reserved = {}

    def available(self, sku):
        return self.on_hand.get(sku, 0) - self.reserved.get(sku, 0)
