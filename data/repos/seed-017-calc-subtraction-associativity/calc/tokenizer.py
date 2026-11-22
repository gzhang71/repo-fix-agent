import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


_TOKEN = re.compile(r"\s*(?:(?P<NUM>\d+(?:\.\d+)?)|(?P<OP>[-+*/])|(?P<LPAREN>\()|(?P<RPAREN>\)))")


def tokenize(text):
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise SyntaxError(f"unexpected character {text[pos]!r} at {pos}")
        kind = m.lastgroup
        tokens.append(Token(kind, m.group(kind)))
        pos = m.end()
    return tokens
