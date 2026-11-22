from webkit.querystring import get_first, parse_qs
from webkit.search import search_terms


def test_repeated_keys():
    assert parse_qs("?tag=a&tag=b&x=1") == {"tag": ["a", "b"], "x": ["1"]}


def test_percent_escapes():
    assert get_first("name=J%C3%BCrgen", "name") == "J\u00fcrgen"


def test_search_terms_split_on_plus():
    assert search_terms("/search?q=Hello+World") == ["hello", "world"]
