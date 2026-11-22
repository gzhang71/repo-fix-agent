from events import EventBus


def test_handlers_run_in_order():
    bus, seen = EventBus(), []
    bus.subscribe("e", lambda p: seen.append(("a", p)))
    bus.subscribe("e", lambda p: seen.append(("b", p)))
    bus.publish("e", 1)
    assert seen == [("a", 1), ("b", 1)]


def test_once_handler_runs_once():
    bus, seen = EventBus(), []
    bus.subscribe("e", seen.append, once=True)
    bus.publish("e", 1); bus.publish("e", 2)
    assert seen == [1]


def test_unsubscribe():
    bus, seen = EventBus(), []
    bus.subscribe("e", seen.append)
    bus.unsubscribe("e", seen.append)
    bus.publish("e", 1)
    assert seen == []
