import re

EMAIL_RE = re.compile(r"^[a-z0-9._+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+$")


def required(value):
    return "is required" if value in (None, "") else None


def email(value):
    if value in (None, ""):
        return None
    return None if EMAIL_RE.match(value) else "is not a valid email"


def length(lo, hi):
    def rule(value):
        if value in (None, ""):
            return None
        return None if lo <= len(value) <= hi else f"must be {lo}-{hi} characters"
    return rule


def no_spaces(value):
    return "must not contain spaces" if value and " " in value else None
