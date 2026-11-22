from tpllint.api import check
from tpllint.lexer import Lexer, Token


def test_tokens_and_lines():
    toks = Lexer().tokenize('x = 1\ny = "a"')
    assert toks[-1] == Token("STRING", '"a"', 2)
    assert [t.kind for t in toks] == ["NAME", "OP", "NUMBER", "NAME", "OP", "STRING"]


def test_check_reports_line():
    assert check("a = 1\nb = @") == ["unexpected '@' at line 2"]


def test_check_clean():
    assert check("{ name | upper }") == []
