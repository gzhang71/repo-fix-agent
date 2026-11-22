from collections import defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass
class Subscription:
    handler: Callable
    once: bool = False


class EventBus:
    """Synchronous pub/sub. Handlers run in subscription order."""

    def __init__(self):
        self._subs = defaultdict(list)

    def subscribe(self, event, handler, once=False):
        self._subs[event].append(Subscription(handler, once))

    def unsubscribe(self, event, handler):
        self._subs[event] = [s for s in self._subs[event] if s.handler != handler]

    def publish(self, event, payload):
        results = []
        for sub in self._subs[event]:
            if sub.once:
                self._subs[event].remove(sub)
            results.append(sub.handler(payload))
        return results
