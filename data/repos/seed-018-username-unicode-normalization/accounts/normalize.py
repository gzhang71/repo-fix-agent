def canonical_username(name):
    """Canonical form used for uniqueness checks: trimmed and case-insensitive."""
    return name.strip().lower()
