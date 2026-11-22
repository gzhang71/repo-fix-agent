import pytest

from orders.order import Order
from orders.serialize import from_dict
from orders.status import OrderStatus


def test_from_dict_lowercase():
    assert from_dict({"id": "3", "status": "shipped"}) == Order(3, OrderStatus.SHIPPED)


def test_advance():
    o = Order(1)
    o.advance("paid")
    assert o.status is OrderStatus.PAID


def test_cannot_advance_final():
    o = Order(1, OrderStatus.CANCELLED)
    with pytest.raises(ValueError, match="already cancelled"):
        o.advance("paid")
