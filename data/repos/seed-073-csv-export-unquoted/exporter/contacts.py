import io

from .csv_export import rows_to_csv


def export_contacts(contacts):
    """Return the CSV text for a list of contact dicts, sorted by id."""
    buf = io.StringIO()
    rows_to_csv(sorted(contacts, key=lambda c: c["id"]), buf)
    return buf.getvalue()
