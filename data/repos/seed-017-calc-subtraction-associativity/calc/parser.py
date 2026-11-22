"""Recursive-descent parser producing tuple ASTs.

    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := NUM | '-' factor | '(' expr ')'
"""


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise SyntaxError(f"unexpected token {self.peek().value!r}")
        return node

    def expr(self):
        left = self.term()
        tok = self.peek()
        if tok is not None and tok.kind == "OP" and tok.value in ("+", "-"):
            self.advance()
            return (tok.value, left, self.expr())
        return left

    def term(self):
        left = self.factor()
        while (tok := self.peek()) is not None and tok.kind == "OP" and tok.value in ("*", "/"):
            self.advance()
            left = (tok.value, left, self.factor())
        return left

    def factor(self):
        tok = self.advance()
        if tok is None:
            raise SyntaxError("unexpected end of expression")
        if tok.kind == "NUM":
            return ("num", float(tok.value) if "." in tok.value else int(tok.value))
        if tok.kind == "OP" and tok.value == "-":
            return ("neg", self.factor())
        if tok.kind == "LPAREN":
            node = self.expr()
            closing = self.advance()
            if closing is None or closing.kind != "RPAREN":
                raise SyntaxError("expected ')'")
            return node
        raise SyntaxError(f"unexpected token {tok.value!r}")
