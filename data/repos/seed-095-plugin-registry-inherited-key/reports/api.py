from exporters import get_exporter


def export_report(rows, fmt="json"):
    exporter = get_exporter(fmt)
    return {"content_type": exporter.content_type, "body": exporter.export(rows)}
