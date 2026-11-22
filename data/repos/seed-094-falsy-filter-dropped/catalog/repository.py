from dataclasses import dataclass

from .query import build_predicate, sort_spec


@dataclass
class Page:
    items: list
    total: int
    page: int
    per_page: int


class ProductRepository:
    def __init__(self, products):
        self._products = list(products)

    def search(self, filters=None, order_by="id", page=1, per_page=20):
        pred = build_predicate(filters or {})
        rows = [p for p in self._products if pred(p)]
        key, reverse = sort_spec(order_by)
        rows.sort(key=key, reverse=reverse)
        start = (page - 1) * per_page
        return Page(rows[start:start + per_page], len(rows), page, per_page)
