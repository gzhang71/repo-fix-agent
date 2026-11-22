COMMANDS = {}


def command(fn):
    """Register fn under its function name; its docstring becomes the help text."""
    COMMANDS[fn.__name__] = fn
    return fn


def help_text():
    return "\n".join(f"{name}: {(fn.__doc__ or '').strip()}" for name, fn in sorted(COMMANDS.items()))


def run(name, *args):
    try:
        fn = COMMANDS[name]
    except KeyError:
        raise SystemExit(f"unknown command: {name}") from None
    return fn(*args)
