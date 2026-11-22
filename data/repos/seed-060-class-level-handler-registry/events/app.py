from .bus import EventBus


def build_buses():
    """Each subsystem gets its own bus so handlers don't cross-talk."""
    return {"orders": EventBus(), "billing": EventBus()}


def wire(buses, log):
    buses["orders"].subscribe("created", lambda p: log.append(("orders", p)))
    buses["billing"].subscribe("created", lambda p: log.append(("billing", p)))
