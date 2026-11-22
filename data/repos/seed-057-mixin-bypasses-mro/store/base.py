class Model:
    """Minimal persistence base. `save()` writes the record to the backing store."""

    table = None

    def __init__(self, store, **fields):
        self.store = store
        self.fields = dict(fields)

    def save(self):
        self.store.setdefault(self.table, []).append(dict(self.fields))
        return self
