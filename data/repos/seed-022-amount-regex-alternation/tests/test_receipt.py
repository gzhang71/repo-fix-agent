from receipts.extract import find_amounts
from receipts.receipt import line_amount, receipt_total


def test_find_simple_amounts():
    assert find_amounts("2 x Bagel 3.00, refund -1.50") == [2.0, 3.0, -1.5]


def test_line_amount_uses_last_number():
    assert line_amount("2 x Filter   3.00") == 3.0
    assert line_amount("THANK YOU") is None


def test_total():
    assert receipt_total(["Tea  2.20", "Cake 3.80", "VISA ****"]) == 6.0
