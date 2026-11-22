_FLAGS = {"new_checkout": False, "dark_mode": False}
_MISSING = object()


def is_enabled(name):
    return bool(_FLAGS.get(name, False))


def set_flag(name, value):
    _FLAGS[name] = bool(value)


def snapshot():
    return dict(_FLAGS)
