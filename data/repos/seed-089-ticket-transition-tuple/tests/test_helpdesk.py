import pytest

from helpdesk.api import ticket_view
from helpdesk.service import InvalidTransition, TicketService


def test_happy_path():
    svc = TicketService()
    t = svc.create("printer on fire")
    for dst in ("open", "pending", "open", "resolved", "closed"):
        svc.move(t.id, dst)
    assert t.status == "closed" and len(t.history) == 5


def test_invalid_transition():
    svc = TicketService()
    t = svc.create("x")
    with pytest.raises(InvalidTransition):
        svc.move(t.id, "resolved")


def test_view_open_ticket():
    svc = TicketService()
    t = svc.create("x")
    svc.move(t.id, "open")
    assert ticket_view(svc, t.id)["actions"] == ["pending", "resolved", "closed"]
