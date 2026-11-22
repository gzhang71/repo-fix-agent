from autocomplete import Trie
from autocomplete.suggest import suggest


def test_contains_and_len():
    t = Trie(["car", "cart", "dog"])
    assert "cart" in t and "car" in t
    assert "ca" not in t
    assert len(t) == 3


def test_suggest_order_and_limit():
    t = Trie(["car", "card", "care", "cat", "dog"])
    assert suggest(t, "Ca", limit=3) == ["car", "cat", "card"]
