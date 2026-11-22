import json

from .serialize import from_dict, to_dict


def dumps(orders):
    return json.dumps([to_dict(o) for o in orders])


def loads(text):
    return [from_dict(d) for d in json.loads(text)]
