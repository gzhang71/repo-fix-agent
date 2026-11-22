from dataclasses import dataclass, field


@dataclass
class Job:
    name: str
    payload: dict = field(default_factory=dict)
