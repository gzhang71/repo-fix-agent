import re


def glob_to_regex(pattern):
    """Translate a glob into an anchored regex over '/'-separated paths.

    '**/' matches zero or more directories, '*' anything except '/', '?' one character.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(pattern[i])
            i += 1
    return re.compile("^" + "".join(out) + "$")
