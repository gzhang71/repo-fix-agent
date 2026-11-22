class Tag:
    """A label attached to articles. Tags are case-insensitive: "Python" == "python"."""

    def __init__(self, name):
        name = name.strip()
        if not name:
            raise ValueError("tag name must not be empty")
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Tag):
            return NotImplemented
        return self.name.lower() == other.name.lower()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Tag({self.name!r})"
