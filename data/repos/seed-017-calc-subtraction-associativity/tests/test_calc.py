import pytest

from calc import calculate


def test_precedence():
    assert calculate("2 + 3 * 4") == 14


def test_parentheses_and_unary():
    assert calculate("(2 + 3) * -4") == -20
    assert calculate("-(1.5 + 0.5)") == -2.0


def test_division_chain():
    assert calculate("8 / 4 / 2") == 1


def test_errors():
    with pytest.raises(ZeroDivisionError):
        calculate("1 / (2 - 2)")
    with pytest.raises(SyntaxError):
        calculate("2 +")
    with pytest.raises(SyntaxError):
        calculate("2 $ 3")
