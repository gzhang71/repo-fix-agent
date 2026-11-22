from .numutil import approx_equal


class Ledger:
    def __init__(self):
        self.entries = []

    def post(self, account, amount):
        self.entries.append((account, float(amount)))

    def balance(self, account=None):
        return sum(amt for acc, amt in self.entries if account is None or acc == account)

    def is_balanced(self):
        return approx_equal(self.balance(), 0.0)
