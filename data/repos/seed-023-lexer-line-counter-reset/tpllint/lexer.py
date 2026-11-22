import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int


class LexError(Exception):
    pass


_SPEC = [
    ("NAME", r"[A-Za-z_]\w*"),
    ("NUMBER", r"\d+"),
    ("STRING", r'"[^"\n]*"'),
    ("OP", r"[=+\-*/(),{}|]"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t]+"),
]
_RX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _SPEC))


class Lexer:
    def __init__(self):
        self.line = 1

    def tokenize(self, text):
        tokens = []
        pos = 0
        while pos < len(text):
            m = _RX.match(text, pos)
            if m is None:
                raise LexError(f"unexpected {text[pos]!r} at line {self.line}")
            kind = m.lastgroup
            if kind == "NEWLINE":
                self.line += 1
            elif kind != "SKIP":
                tokens.append(Token(kind, m.group(), self.line))
            pos = m.end()
        return tokens
