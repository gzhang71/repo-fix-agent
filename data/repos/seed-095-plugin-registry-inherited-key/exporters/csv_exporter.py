import csv
import io

from .base import Exporter


class CsvExporter(Exporter):
    format = "csv"
    content_type = "text/csv"

    def export(self, rows):
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
