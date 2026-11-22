from .decorators import counted
from .registry import command


@command
def greet(name):
    """Say hello."""
    return f"hello {name}"


@command
@counted
def deploy(env):
    """Deploy the current build."""
    return f"deployed to {env}"


@command
@counted
def rollback(env):
    """Roll back the last deploy."""
    return f"rolled back {env}"
