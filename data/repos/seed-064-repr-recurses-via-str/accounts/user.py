class User:
    """An account holder.

    str(user)  -> the display name, e.g. "Ada Lovelace"
    repr(user) -> "<User Ada Lovelace id=7>"
    """

    def __init__(self, id, first, last, email):
        self.id = id
        self.first = first
        self.last = last
        self.email = email

    @property
    def display_name(self):
        return f"{self.first} {self.last}".strip()

    def __repr__(self):
        return f"<User {self} id={self.id}>"
