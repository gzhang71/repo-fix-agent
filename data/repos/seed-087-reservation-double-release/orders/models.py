from dataclasses import dataclass, field


@dataclass
class Order:
    id: str
    items: dict = field(default_factory=dict)
    status: str = "pending"
