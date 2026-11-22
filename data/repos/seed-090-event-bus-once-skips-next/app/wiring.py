from events import EventBus

from .audit import AuditLog
from .bootstrap import Bootstrap
from .mailer import Mailer


class App:
    def __init__(self):
        self.bus = EventBus()
        self.audit = AuditLog()
        self.bootstrap = Bootstrap()
        self.mailer = Mailer()
        self.users = []
        self.bus.subscribe("user.created", self.audit.record)
        self.bus.subscribe("user.created", self.bootstrap.make_first_user_admin, once=True)
        self.bus.subscribe("user.created", self.mailer.send_welcome)

    def signup(self, email):
        user = {"email": email, "role": "member"}
        self.users.append(user)
        self.bus.publish("user.created", user)
        return user
