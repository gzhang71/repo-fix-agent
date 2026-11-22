class UnknownFilter(KeyError):
    pass


OPERATORS = {
    "category": lambda p, v: p.category == v,
    "in_stock": lambda p, v: p.in_stock == v,
    "min_price": lambda p, v: p.price_cents >= v,
    "max_price": lambda p, v: p.price_cents <= v,
    "min_rating": lambda p, v: p.rating >= v,
}

SORTS = {
    "id": (lambda p: p.id, False),
    "price": (lambda p: (p.price_cents, p.id), False),
    "-price": (lambda p: (-p.price_cents, p.id), False),
    "-rating": (lambda p: (-p.rating, p.id), False),
    "name": (lambda p: (p.name.lower(), p.id), False),
}


def build_predicate(filters):
    checks = []
    for key, value in filters.items():
        if not value:  # unset filter
            continue
        if key not in OPERATORS:
            raise UnknownFilter(key)
        checks.append((OPERATORS[key], value))
    return lambda product: all(fn(product, v) for fn, v in checks)


def sort_spec(order_by):
    try:
        return SORTS[order_by]
    except KeyError:
        raise ValueError(f"cannot sort by {order_by!r}") from None
