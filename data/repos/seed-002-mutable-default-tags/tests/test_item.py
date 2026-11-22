from inventory import Item


def test_add_tag_dedups():
    it = Item("pear", tags=["fruit"])
    it.add_tag("fruit")
    it.add_tag("green")
    assert it.tags == ["fruit", "green"]
