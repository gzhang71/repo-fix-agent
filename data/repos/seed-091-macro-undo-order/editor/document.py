class Document:
    def __init__(self, text=""):
        self.text = text

    def __repr__(self):
        return f"Document({self.text!r})"
