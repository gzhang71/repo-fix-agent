class Cart:
    def __init__(self):
        self.lines = []

    def add(self, product, qty=1):
        if qty > product.stock:
            raise ValueError(f"only {product.stock} of {product.sku} in stock")
        self.lines.append((product, qty))

    def total(self):
        return sum(p.price * qty for p, qty in self.lines)
