class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__(errors)
        self.errors = errors


class Pipeline:
    """Normalise input with `cleaners`, then check the cleaned values with `validators`.

    Returns the cleaned dict or raises ValidationError({field: [messages]}).
    """

    def __init__(self, cleaners, validators):
        self.cleaners = cleaners
        self.validators = validators

    def run(self, data):
        errors = {}
        for field, rules in self.validators.items():
            value = data.get(field)
            for rule in rules:
                msg = rule(value)
                if msg:
                    errors.setdefault(field, []).append(msg)
        cleaned = dict(data)
        for field, fns in self.cleaners.items():
            if cleaned.get(field) is not None:
                for fn in fns:
                    cleaned[field] = fn(cleaned[field])
        if errors:
            raise ValidationError(errors)
        return cleaned
