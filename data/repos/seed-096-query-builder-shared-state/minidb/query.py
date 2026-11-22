import operator

LOOKUPS = {"eq": operator.eq, "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}


def _parse(key):
    field, _, op = key.partition("__")
    return field, LOOKUPS[op or "eq"]


class Query:
    def __init__(self, table):
        self.table = table
        self._filters = []
        self._order = None
        self._limit = None

    def where(self, **conds):
        for key, value in conds.items():
            field, op = _parse(key)
            self._filters.append((field, op, value))
        return self

    def order_by(self, field):
        self._order = field
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _matching(self):
        return [r for r in self.table.rows
                if all(op(r.get(f), v) for f, op, v in self._filters)]

    def all(self):
        rows = self._matching()
        if self._order:
            field = self._order.lstrip("-")
            rows.sort(key=lambda r: r[field], reverse=self._order.startswith("-"))
        return rows[:self._limit] if self._limit is not None else rows

    def count(self):
        return len(self._matching())

    def first(self):
        rows = self.limit(1).all()
        return rows[0] if rows else None
