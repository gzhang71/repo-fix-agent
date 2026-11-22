from .paths import is_inside, relative_to_root


def plan_upload(root, changed_paths):
    """Root-relative paths of the changed files that live under `root`, sorted."""
    return sorted(relative_to_root(root, p) for p in changed_paths if is_inside(root, p))
