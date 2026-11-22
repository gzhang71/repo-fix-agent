import pytest

from catalog.cart import Cart
from catalog.product import Product


def test_negative_price_rejected():
    with pytest.raises(ValueError, match="price"):
        Product("X", price=-1)


def test_single_product_cart():
    p = Product("MUG", price=5, stock=10)
    cart = Cart()
    cart.add(p, 3)
    assert cart.total() == 15


def test_out_of_stock():
    p = Product("MUG", price=5, stock=1)
    with pytest.raises(ValueError, match="only 1"):
        Cart().add(p, 2)
