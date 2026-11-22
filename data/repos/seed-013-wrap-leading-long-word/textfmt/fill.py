from .wrap import wrap


def fill(text, width=72, indent=""):
    """Wrap text to `width` columns, prefixing every line with `indent`."""
    inner = width - len(indent)
    if inner < 1:
        raise ValueError("indent is wider than width")
    return "\n".join(indent + line for line in wrap(text, inner))
