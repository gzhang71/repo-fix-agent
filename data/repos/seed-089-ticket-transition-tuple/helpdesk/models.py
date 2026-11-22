from dataclasses import dataclass, field


@dataclass
class Ticket:
    id: int
    title: str
    status: str = "new"
    history: list = field(default_factory=list)
