class FakeGateway:
    def __init__(self):
        self.charges = []
        self.refunds = []

    def charge(self, order_id, cents):
        self.charges.append((order_id, cents))

    def refund(self, order_id, cents):
        self.refunds.append((order_id, cents))
