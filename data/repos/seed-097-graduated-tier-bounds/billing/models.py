from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tier:
    up_to: Optional[int]  # inclusive bound on *cumulative* units; None = no limit
    unit_cents: int


@dataclass(frozen=True)
class Plan:
    name: str
    base_cents: int
    tiers: tuple
    mode: str = "graduated"  # or "volume"
