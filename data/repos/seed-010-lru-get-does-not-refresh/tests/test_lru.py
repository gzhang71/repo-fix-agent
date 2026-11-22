from cache import LRUCache


def test_evicts_oldest_insert():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None and c.get("c") == 3
    assert len(c) == 2


def test_put_refreshes():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3)
    assert c.get("a") == 10 and c.get("b") is None
