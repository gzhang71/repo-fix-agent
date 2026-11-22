import pytest

from inventory.stock import Stock
from orders.models import Order
from orders.service import CannotReturn, OrderService
from orders.status import OrderStatus
from payments.gateway import FakeGateway


def make():
    svc = OrderService(Stock({"MUG": 5}), FakeGateway())
    svc.create(Order(1042, {"MUG": 2}, 4000))
    return svc


def test_cancel_paid_order_refunds_and_restocks():
    svc = make()
    svc.pay(1042)
    svc.cancel(1042)
    assert svc.orders[1042].status is OrderStatus.CANCELLED
    assert svc.gateway.refunds == [(1042, 4000)]
    assert svc.stock.levels["MUG"] == 5


def test_cancel_pending_order_no_refund():
    svc = make()
    svc.cancel(1042)
    assert svc.gateway.refunds == []
    assert svc.stock.levels["MUG"] == 5


def test_cannot_return_undelivered():
    svc = make()
    svc.pay(1042)
    with pytest.raises(CannotReturn):
        svc.return_order(1042)
