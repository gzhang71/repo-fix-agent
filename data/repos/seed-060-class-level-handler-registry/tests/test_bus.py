from events.bus import EventBus


def test_publish_calls_handlers_in_order():
    bus, seen = EventBus(), []
    bus.subscribe("ping", lambda p: seen.append(("first", p)))
    bus.subscribe("ping", lambda p: seen.append(("second", p)))
    assert bus.publish("ping", 1) == 2
    assert seen == [("first", 1), ("second", 1)]


def test_publish_without_handlers():
    assert EventBus().publish("nobody-listens") == 0
