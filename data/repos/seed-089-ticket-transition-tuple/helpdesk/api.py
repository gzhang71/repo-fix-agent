from .states import next_states


def ticket_view(service, ticket_id):
    t = service.tickets[ticket_id]
    return {"id": t.id, "title": t.title, "status": t.status,
            "actions": next_states(t.status)}
