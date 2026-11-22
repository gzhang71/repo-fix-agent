import pytest

from shop.checkout import checkout, receipt
from shop.coupons import CouponError, discount_for
from shop.models import LineItem


def test_no_coupon_ca():
    order = checkout([LineItem("mug", 2000, 2)], "CA")
    assert receipt(order) == {"subtotal": 4000, "discount": 0, "shipping": 599,
                              "tax": 290, "total": 4889}


def test_coupon_amounts():
    assert discount_for("save10", 8000) == 800
    assert discount_for("FIVEOFF", 3000) == 500


def test_coupon_minimum():
    with pytest.raises(CouponError):
        discount_for("FIVEOFF", 1000)


def test_free_shipping_threshold():
    assert checkout([LineItem("lamp", 5000)], "OR").shipping_cents == 0
