import json
from pathlib import Path


def save_state(state, path):
    Path(path).write_text(json.dumps(state, sort_keys=True))


def load_state(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
