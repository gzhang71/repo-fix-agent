from ledger.book import Ledger
from ledger.numutil import approx_equal


def test_relative_tolerance():
    assert approx_equal(1e10, 1e10 + 1)
    assert not approx_equal(1.0, 1.1)


def test_balance_per_account():
    lg = Ledger()
    lg.post("cash", 10)
    lg.post("sales", -10)
    lg.post("cash", 5)
    assert lg.balance("cash") == 15
    assert lg.balance() == 5


def test_exact_balance():
    lg = Ledger()
    lg.post("cash", 25)
    lg.post("sales", -25)
    assert lg.is_balanced()


def test_real_imbalance_detected():
    lg = Ledger()
    lg.post("cash", 10.00)
    lg.post("sales", -9.99)
    assert not lg.is_balanced()
