class History:
    def __init__(self, doc):
        self.doc = doc
        self._undo = []
        self._redo = []

    def execute(self, cmd):
        cmd.do(self.doc)
        self._undo.append(cmd)
        self._redo.clear()

    def undo(self):
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(self.doc)
        self._redo.append(cmd)
        return True

    def redo(self):
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.do(self.doc)
        self._undo.append(cmd)
        return True
