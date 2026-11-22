from .lexer import Lexer, LexError

_lexer = Lexer()


def tokenize(text):
    return _lexer.tokenize(text)


def check(text):
    """Return a list of error messages (empty if the template lexes cleanly)."""
    try:
        tokenize(text)
    except LexError as e:
        return [str(e)]
    return []


def check_all(files):
    """files: {name: text}. Returns ["name: message", ...]."""
    return [f"{name}: {msg}" for name, text in files.items() for msg in check(text)]
