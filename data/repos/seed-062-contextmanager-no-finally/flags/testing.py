from contextlib import contextmanager

from . import registry


@contextmanager
def override_flags(**overrides):
    """Temporarily set feature flags, restoring the previous values afterwards."""
    previous = {name: registry._FLAGS.get(name, registry._MISSING) for name in overrides}
    for name, value in overrides.items():
        registry.set_flag(name, value)
    yield
    for name, old in previous.items():
        if old is registry._MISSING:
            del registry._FLAGS[name]
        else:
            registry._FLAGS[name] = old
