import json

from .json_exporter import JsonExporter


class PrettyJsonExporter(JsonExporter):
    """Indented JSON for the internal debug console. Not an API format."""

    def export(self, rows):
        return json.dumps(rows, indent=2, sort_keys=True)


def dump_for_console(rows):
    return PrettyJsonExporter().export(rows)
