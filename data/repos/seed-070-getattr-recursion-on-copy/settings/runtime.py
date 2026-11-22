import copy

from .proxy import SettingsProxy

BASE = SettingsProxy({"debug": False, "workers": 4}, defaults={"timeout": 30})


def per_request(settings=BASE):
    """Give each request its own copy so handlers can't affect each other."""
    return copy.deepcopy(settings)
