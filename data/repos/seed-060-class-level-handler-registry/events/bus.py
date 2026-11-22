class EventBus:
    """Simple synchronous publish/subscribe bus."""

    _handlers = {}

    def subscribe(self, event, handler):
        self._handlers.setdefault(event, []).append(handler)

    def publish(self, event, payload=None):
        handlers = self._handlers.get(event, [])
        for handler in handlers:
            handler(payload)
        return len(handlers)
