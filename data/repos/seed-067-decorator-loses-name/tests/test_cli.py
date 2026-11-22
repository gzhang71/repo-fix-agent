from cli import commands  # noqa: F401  (registers commands)
from cli.decorators import counted
from cli.registry import run


def test_greet():
    assert run("greet", "bob") == "hello bob"


def test_counted_counts_calls():
    @counted
    def f(x):
        return x * 2

    assert f(2) == 4 and f(3) == 6
    assert f.calls == 2
