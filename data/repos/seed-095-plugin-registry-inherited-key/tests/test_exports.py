import json

import pytest

from exporters import UnknownFormat
from exporters.debug import dump_for_console
from reports.api import export_report

ROWS = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_json_roundtrip():
    out = export_report(ROWS, "json")
    assert out["content_type"] == "application/json"
    assert json.loads(out["body"]) == ROWS


def test_csv():
    assert export_report(ROWS, "csv")["body"] == "id,name\n1,a\n2,b\n"


def test_unknown_format():
    with pytest.raises(UnknownFormat):
        export_report(ROWS, "xml")


def test_debug_console_is_indented():
    assert dump_for_console([{"a": 1}]) == '[\n  {\n    "a": 1\n  }\n]'
