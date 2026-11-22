import pytest

from checkout.cart import Cart
from checkout.pricing import price
from promo.coupons import InvalidCoupon


def cart_with(*codes):
    cart = Cart()
    cart.add("shoes", 6000)
    cart.add("socks", 2000, 2)
    for code in codes:
        cart.add_code(code)
    return cart


def test_no_coupons():
    assert price(cart_with())["total"] == 10000


def test_non_exclusive_stack():
    out = price(cart_with("WELCOME5", "spring10"))
    assert out["applied"] == ["WELCOME5", "SPRING10"]
    assert out["total"] == 8500


def test_exclusive_first_blocks_others():
    out = price(cart_with("VIP20", "WELCOME5"))
    assert out["applied"] == ["VIP20"] and out["total"] == 8000


def test_duplicates_ignored():
    assert price(cart_with("WELCOME5", "WELCOME5"))["total"] == 9500


def test_invalid_code():
    with pytest.raises(InvalidCoupon):
        cart_with("BOGUS")
