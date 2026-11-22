from quotes.pricing import quote


CART = [["sku-1", 2, 500], ["sku-2", 1, 1000]]


def test_quote_us():
    quote.cache_clear()
    assert quote(CART) == 2140


def test_list_args_are_cached():
    quote.cache_clear()
    quote(CART)
    quote([list(row) for row in CART])
    assert len(quote.cache) == 1
