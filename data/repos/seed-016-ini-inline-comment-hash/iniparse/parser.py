class ParseError(ValueError):
    pass


def _strip_comment(line):
    """Drop a trailing inline comment.

    Inside a value, an inline comment starts at a '#' or ';' preceded by whitespace,
    so values such as `#ff8800` or `a;b` are kept intact.
    """
    for marker in ("#", ";"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line


def parse_ini(text):
    data, section = {}, None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            raise ParseError(f"line {lineno}: expected 'key = value'")
        if section is None:
            raise ParseError(f"line {lineno}: key outside of a section")
        key, _, value = line.partition("=")
        data[section][key.strip().lower()] = value.strip()
    return data
