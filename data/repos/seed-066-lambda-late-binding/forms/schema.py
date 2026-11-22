from .validators import max_length, required

FIELDS = {
    "username": 20,
    "email": 254,
    "display_name": 50,
}


def build_rules(fields=None):
    """Return {field: [callable(value) -> error message or None]}."""
    rules = {}
    for name, limit in (fields or FIELDS).items():
        rules[name] = [
            lambda value: required(name, value),
            lambda value: max_length(limit)(name, value),
        ]
    return rules


def validate(data, rules=None):
    rules = rules or build_rules()
    errors = {}
    for name, checks in rules.items():
        for check in checks:
            msg = check(data.get(name))
            if msg:
                errors[name] = msg
                break
    return errors
