def required(field, value):
    if value in (None, ""):
        return f"{field} is required"
    return None


def max_length(limit):
    def check(field, value):
        if value is not None and len(value) > limit:
            return f"{field} must be at most {limit} characters"
        return None
    return check
