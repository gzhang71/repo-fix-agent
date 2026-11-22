from dataclasses import dataclass


@dataclass(order=True)
class Job:
    """A unit of work. Lower `priority` runs first; ties run in submission order (`seq`)."""

    name: str
    priority: int
    seq: int = 0
