import re

_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name):
    """userName -> user_name"""
    return _BOUNDARY.sub("_", name).lower()


def snake_to_camel(name):
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)
