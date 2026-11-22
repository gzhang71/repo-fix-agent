import copy


class InvoiceTemplate:
    def __init__(self, name, lines=None, meta=None):
        self.name = name
        self.lines = list(lines or [])  # [{"desc": str, "amount": int}]
        self.meta = dict(meta or {})

    def clone(self, **changes):
        """Return an independent copy, optionally overriding attributes."""
        new = copy.copy(self)
        for key, value in changes.items():
            setattr(new, key, value)
        return new

    def total(self):
        return sum(line["amount"] for line in self.lines)
