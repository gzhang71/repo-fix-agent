from .durations import format_duration


def describe_run(name, started, finished):
    return f"{name} took {format_duration(finished - started)}"
