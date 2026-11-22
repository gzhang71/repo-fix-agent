import pytest

from checkout.place import place_order
from orders.models import Line
from payments.ledger import Ledger, RefundError
from pricing.allocation import allocate
from refunds.service import AlreadyRefunded, refund_line


def test_allocate_even_split():
    assert allocate(600, [100, 200, 300]) == [100, 200, 300]
    assert allocate(0, [5, 5]) == [0, 0]
    assert allocate(10, [0, 0]) == [0, 0]


def test_refund_single_line():
    ledger = Ledger()
    order = place_order(1, [Line("a", 1000), Line("b", 3000)], ledger, discount_cents=400)
    assert refund_line(order, 1, ledger) == 2700
    assert ledger.remaining(1) == 900


def test_cannot_refund_twice():
    ledger = Ledger()
    order = place_order(1, [Line("a", 1000)], ledger)
    refund_line(order, 0, ledger)
    with pytest.raises(AlreadyRefunded):
        refund_line(order, 0, ledger)


def test_ledger_rejects_over_refund():
    ledger = Ledger()
    ledger.capture(9, 100)
    with pytest.raises(RefundError):
        ledger.refund(9, 101)
