_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def unescape(s):
    """Decode backslash escapes: \\n, \\t, \\r; any other escaped char stands for itself."""
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)
