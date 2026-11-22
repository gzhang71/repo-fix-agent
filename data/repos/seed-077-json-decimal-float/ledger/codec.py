import json
from decimal import Decimal


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def dump_entries(entries):
    return json.dumps({"entries": entries}, cls=_Encoder)


def load_entries(text):
    return json.loads(text)["entries"]
