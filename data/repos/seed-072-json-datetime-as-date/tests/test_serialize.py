import io
from datetime import date

from eventlog.serialize import dumps, loads
from eventlog.store import append, read_all


def test_date_round_trip():
    ev = {"day": date(2024, 2, 29), "kind": "close"}
    assert loads(dumps(ev)) == ev


def test_set_round_trip():
    ev = {"tags": {"b", "a"}}
    assert loads(dumps(ev)) == ev
    assert '"__set__": ["a", "b"]' in dumps(ev)


def test_store_lines():
    buf = io.StringIO()
    append(buf, {"n": 1})
    append(buf, {"n": 2})
    assert read_all(buf) == [{"n": 1}, {"n": 2}]
