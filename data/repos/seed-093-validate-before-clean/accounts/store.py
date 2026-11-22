class DuplicateUser(Exception):
    pass


class UserStore:
    def __init__(self):
        self.users = {}

    def create(self, data):
        if data["email"] in self.users:
            raise DuplicateUser(data["email"])
        self.users[data["email"]] = dict(data)
        return self.users[data["email"]]
