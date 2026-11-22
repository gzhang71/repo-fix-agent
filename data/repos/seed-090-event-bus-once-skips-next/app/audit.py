class AuditLog:
    def __init__(self):
        self.entries = []

    def record(self, user):
        self.entries.append(("user.created", user["email"]))
