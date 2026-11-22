import json

from .base import Exporter


class JsonExporter(Exporter):
    format = "json"
    content_type = "application/json"

    def export(self, rows):
        return json.dumps(rows, separators=(",", ":"), sort_keys=True)
