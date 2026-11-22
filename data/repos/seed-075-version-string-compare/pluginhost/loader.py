from .versions import satisfies


def load_plugins(manifests, host_version):
    """Split plugin manifests into (loaded names, {name: reason} for skipped)."""
    loaded, skipped = [], {}
    for m in manifests:
        need = m.get("requires_host", "0")
        if satisfies(host_version, need):
            loaded.append(m["name"])
        else:
            skipped[m["name"]] = f"requires host >= {need}"
    return loaded, skipped
