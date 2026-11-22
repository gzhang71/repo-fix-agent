import json
from pathlib import Path


class DiskCache:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.dir / f"{key}.json"

    def get(self, key):
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def set(self, key, value):
        self._path(key).write_text(json.dumps(value))
