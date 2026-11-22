from .iterutils import chunked, dedupe


def normalize_email(row):
    return row["email"].strip().lower()


def import_contacts(rows):
    """Contacts de-duplicated by normalized email; the first occurrence wins."""
    return list(dedupe(rows, key=normalize_email))


def batches(rows, size=100):
    return list(chunked(import_contacts(rows), size))
