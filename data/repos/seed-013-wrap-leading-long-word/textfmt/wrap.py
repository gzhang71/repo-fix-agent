def wrap(text, width):
    """Greedy word wrap into lines of at most `width` characters.

    Words longer than `width` are never split; they go on a line of their own.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    lines, current = [], ""
    for word in text.split():
        needed = len(word) + (1 if current else 0)
        if len(current) + needed > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines
