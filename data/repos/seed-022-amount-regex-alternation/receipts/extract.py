import re

# 1234.50, 1,234.50, -3, 12
_AMOUNT = re.compile(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?")


def find_amounts(text):
    """All monetary amounts in `text`, in order, as floats."""
    return [float(m.group().replace(",", "")) for m in _AMOUNT.finditer(text)]
