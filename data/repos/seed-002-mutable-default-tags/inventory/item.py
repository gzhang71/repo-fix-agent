class Item:
    def __init__(self, name, tags=[]):
        self.name = name
        self.tags = tags

    def add_tag(self, tag):
        if tag not in self.tags:
            self.tags.append(tag)

    def __repr__(self):
        return f"Item({self.name!r}, tags={self.tags!r})"
