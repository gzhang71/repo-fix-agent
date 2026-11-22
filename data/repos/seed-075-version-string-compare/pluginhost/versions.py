def normalize(version):
    return version.strip().lstrip("v")


def satisfies(installed, minimum):
    """True if version `installed` is >= `minimum` (dotted numeric versions)."""
    return normalize(installed) >= normalize(minimum)
