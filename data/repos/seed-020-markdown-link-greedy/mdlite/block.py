from .inline import render_inline


def render(source):
    """Render headings (#..###) and paragraphs separated by blank lines."""
    out = []
    for block in source.strip().split("\n\n"):
        block = " ".join(line.strip() for line in block.splitlines())
        if not block:
            continue
        level = len(block) - len(block.lstrip("#"))
        if 1 <= level <= 3 and block[level:level + 1] == " ":
            out.append(f"<h{level}>{render_inline(block[level + 1:])}</h{level}>")
        else:
            out.append(f"<p>{render_inline(block)}</p>")
    return "\n".join(out)
