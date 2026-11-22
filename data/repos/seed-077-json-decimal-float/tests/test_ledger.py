from decimal import Decimal

from ledger.codec import dump_entries, load_entries
from ledger.report import balance


def test_dump_writes_numbers():
    text = dump_entries([{"memo": "coffee", "amount": Decimal("3.50")}])
    assert text == '{"entries": [{"memo": "coffee", "amount": 3.5}]}'


def test_balance_whole_amounts():
    text = dump_entries([{"amount": 10}, {"amount": -3}])
    assert balance(text) == Decimal("7.00")


def test_load_keeps_other_fields():
    assert load_entries('{"entries": [{"memo": "x", "amount": 1}]}') == [{"memo": "x", "amount": 1}]
