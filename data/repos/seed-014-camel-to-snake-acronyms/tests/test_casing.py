from apiclient.casing import camel_to_snake, snake_to_camel
from apiclient.payload import normalize_keys


def test_simple_camel():
    assert camel_to_snake("displayName") == "display_name"
    assert camel_to_snake("Simple") == "simple"
    assert camel_to_snake("already_snake") == "already_snake"


def test_snake_to_camel():
    assert snake_to_camel("display_name") == "displayName"


def test_normalize_nested():
    raw = {"pageInfo": {"nextCursor": "x"}, "items": [{"itemName": "a"}]}
    assert normalize_keys(raw) == {"page_info": {"next_cursor": "x"}, "items": [{"item_name": "a"}]}
