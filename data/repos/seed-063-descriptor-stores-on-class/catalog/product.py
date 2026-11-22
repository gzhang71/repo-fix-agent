from .fields import PositiveNumber


class Product:
    price = PositiveNumber()
    stock = PositiveNumber()

    def __init__(self, sku, price, stock=0):
        self.sku = sku
        self.price = price
        self.stock = stock

    def __repr__(self):
        return f"Product({self.sku!r}, price={self.price!r}, stock={self.stock!r})"
