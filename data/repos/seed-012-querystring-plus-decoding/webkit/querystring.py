from urllib.parse import unquote


def parse_qs(qs):
    """Parse an application/x-www-form-urlencoded query string into {key: [values]}."""
    if qs.startswith("?"):
        qs = qs[1:]
    out = {}
    for pair in qs.split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        key = unquote(key).replace("+", " ")
        value = unquote(value).replace("+", " ")
        out.setdefault(key, []).append(value)
    return out


def get_first(qs, key, default=None):
    return parse_qs(qs).get(key, [default])[0]
