from .query import Query


class Table:
    def __init__(self, name):
        self.name = name
        self.rows = []
        self._next_id = 1

    def insert(self, **fields):
        row = {"id": self._next_id, **fields}
        self._next_id += 1
        self.rows.append(row)
        return row

    def query(self):
        return Query(self)
