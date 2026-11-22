TRANSITIONS = {
    "new": ("open", "closed"),
    "open": ("pending", "resolved", "closed"),
    "pending": ("open"),
    "resolved": ("open", "closed"),
    "closed": (),
}


def can_transition(src, dst):
    return dst in TRANSITIONS[src]


def next_states(src):
    return list(TRANSITIONS[src])
