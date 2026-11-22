def split_line(line, sep=","):
    """Split one CSV record into fields.

    Fields may be wrapped in double quotes so they can contain `sep`; a literal
    quote inside a quoted field is written as two quotes ("").
    """
    fields, buf, in_quotes = [], [], False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == sep and not in_quotes:
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    fields.append("".join(buf))
    return fields
