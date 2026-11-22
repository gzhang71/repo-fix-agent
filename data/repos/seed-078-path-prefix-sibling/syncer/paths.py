from pathlib import PurePosixPath


def is_inside(root, path):
    """True if `path` is `root` itself or somewhere beneath it."""
    return str(PurePosixPath(path)).startswith(str(PurePosixPath(root)))


def relative_to_root(root, path):
    return str(PurePosixPath(path).relative_to(root))
