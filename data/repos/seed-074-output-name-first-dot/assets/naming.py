def output_name(src, new_ext):
    """Replace the extension of `src`: ('img/cat.png', '.webp') -> 'img/cat.webp'."""
    base = src.split(".")[0]
    return base + new_ext
