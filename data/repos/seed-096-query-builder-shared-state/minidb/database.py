from .table import Table


class Database:
    def __init__(self):
        self._tables = {}

    def table(self, name):
        if name not in self._tables:
            self._tables[name] = Table(name)
        return self._tables[name]
