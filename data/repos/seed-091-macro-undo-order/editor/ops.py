from .commands import Delete, Insert, Macro


def find_all(text, needle):
    out, i = [], text.find(needle)
    while needle and i != -1:
        out.append(i)
        i = text.find(needle, i + len(needle))
    return out


def type_text(history, pos, text):
    history.execute(Insert(pos, text))


def delete_range(history, start, end):
    history.execute(Delete(start, end - start))


def wrap_selection(history, start, end, before, after):
    """E.g. bold: wrap_selection(h, 0, 5, "**", "**")."""
    history.execute(Macro([Insert(end, after), Insert(start, before)], label="wrap"))


def replace_all(history, old, new):
    positions = find_all(history.doc.text, old)
    if not positions:
        return 0
    cmds = []
    for pos in reversed(positions):  # right to left keeps earlier offsets valid
        cmds += [Delete(pos, len(old)), Insert(pos, new)]
    history.execute(Macro(cmds, label=f"replace {old!r}"))
    return len(positions)
