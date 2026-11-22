from pathlib import Path

from .fingerprint import fingerprint


class Builder:
    def __init__(self, compile_fn, state=None):
        self.compile_fn = compile_fn
        self.state = state if state is not None else {}  # path -> [fingerprint, output]
        self.compiled = 0

    def build(self, path):
        fp = fingerprint(path)
        entry = self.state.get(str(path))
        if entry and entry[0] == fp:
            return entry[1]
        out = self.compile_fn(Path(path).read_text())
        self.compiled += 1
        self.state[str(path)] = [fp, out]
        return out
