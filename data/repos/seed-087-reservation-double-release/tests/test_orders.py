import pytest

from orders.service import OrderService
from warehouse.reservations import OutOfStock, ReservationBook
from warehouse.stock import Stock


def make():
    stock = Stock({"TEE-M": 10})
    return stock, OrderService(ReservationBook(stock))


def test_place_reserves():
    stock, svc = make()
    svc.place("o1", {"TEE-M": 3}, now=0)
    assert stock.available("TEE-M") == 7


def test_cannot_oversell():
    stock, svc = make()
    with pytest.raises(OutOfStock):
        svc.place("o1", {"TEE-M": 11}, now=0)


def test_ship_decrements_on_hand():
    stock, svc = make()
    svc.place("o1", {"TEE-M": 4}, now=0)
    svc.ship("o1")
    assert stock.on_hand["TEE-M"] == 6 and stock.available("TEE-M") == 6


def test_sweep_expires_stale_holds():
    stock, svc = make()
    svc.place("o1", {"TEE-M": 2}, now=0)
    svc.sweep(now=100)
    assert svc.orders["o1"].status == "expired"
    assert stock.available("TEE-M") == 10
