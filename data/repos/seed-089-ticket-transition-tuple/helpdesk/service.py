from .models import Ticket
from .states import can_transition


class InvalidTransition(Exception):
    pass


class TicketService:
    def __init__(self):
        self.tickets = {}
        self._next = 1

    def create(self, title):
        t = Ticket(self._next, title)
        self.tickets[t.id] = t
        self._next += 1
        return t

    def move(self, ticket_id, dst):
        t = self.tickets[ticket_id]
        if not can_transition(t.status, dst):
            raise InvalidTransition(f"{t.status} -> {dst}")
        t.history.append((t.status, dst))
        t.status = dst
        return t
