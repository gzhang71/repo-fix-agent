import re

from .escape import escape_html

_VAR = re.compile(r"\{\{\s*(\w+)(\|safe)?\s*\}\}")


def render(template, context):
    """Substitute {{ name }} (HTML-escaped) and {{ name|safe }} (raw)."""
    def sub(m):
        name, safe = m.group(1), m.group(2)
        if name not in context:
            raise KeyError(f"template variable {name!r} is not defined")
        value = context[name]
        return str(value) if safe else escape_html(value)
    return _VAR.sub(sub, template)
