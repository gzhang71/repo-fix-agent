class Mailer:
    def __init__(self):
        self.outbox = []

    def send_welcome(self, user):
        self.outbox.append((user["email"], "Welcome aboard!"))
