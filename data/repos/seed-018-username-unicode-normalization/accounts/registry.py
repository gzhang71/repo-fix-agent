from .normalize import canonical_username


class UsernameTaken(ValueError):
    pass


class Registry:
    def __init__(self):
        self._users = {}

    def register(self, name):
        key = canonical_username(name)
        if not key:
            raise ValueError("username must not be blank")
        if key in self._users:
            raise UsernameTaken(name)
        self._users[key] = name
        return key

    def is_taken(self, name):
        return canonical_username(name) in self._users
