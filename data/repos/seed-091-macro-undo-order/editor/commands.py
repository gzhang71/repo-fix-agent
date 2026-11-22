class Insert:
    def __init__(self, pos, text):
        self.pos, self.text = pos, text

    def do(self, doc):
        doc.text = doc.text[:self.pos] + self.text + doc.text[self.pos:]

    def undo(self, doc):
        doc.text = doc.text[:self.pos] + doc.text[self.pos + len(self.text):]


class Delete:
    def __init__(self, pos, length):
        self.pos, self.length = pos, length
        self.removed = None

    def do(self, doc):
        self.removed = doc.text[self.pos:self.pos + self.length]
        doc.text = doc.text[:self.pos] + doc.text[self.pos + self.length:]

    def undo(self, doc):
        doc.text = doc.text[:self.pos] + self.removed + doc.text[self.pos:]


class Macro:
    """Several commands applied (and undone) as one history entry."""

    def __init__(self, commands, label="macro"):
        self.commands = list(commands)
        self.label = label

    def do(self, doc):
        for cmd in self.commands:
            cmd.do(doc)

    def undo(self, doc):
        for cmd in self.commands:
            cmd.undo(doc)
