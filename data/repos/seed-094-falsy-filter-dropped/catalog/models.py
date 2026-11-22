from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    category: str
    price_cents: int
    in_stock: bool
    rating: int
