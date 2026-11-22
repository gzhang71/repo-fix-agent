import json
from datetime import date, datetime


class EventEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, date):
            return {"__date__": obj.isoformat()}
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, (set, frozenset)):
            return {"__set__": sorted(obj)}
        return super().default(obj)


def dumps(event):
    return json.dumps(event, cls=EventEncoder, sort_keys=True)


def _hook(obj):
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    if "__date__" in obj:
        return date.fromisoformat(obj["__date__"])
    if "__set__" in obj:
        return set(obj["__set__"])
    return obj


def loads(text):
    return json.loads(text, object_hook=_hook)
