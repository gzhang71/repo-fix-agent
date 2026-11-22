from billing.catalog import get_plan
from billing.invoice import build_invoice, format_cents
from billing.tiers import usage_cost
from billing.usage import total_units

EVENTS = [("acme", "2024-10", 2000), ("acme", "2024-10", 1000),
          ("acme", "2024-09", 50000), ("globex", "2024-10", 700)]


def test_total_units():
    assert total_units(EVENTS, "acme", "2024-10") == 3000


def test_free_tier():
    assert usage_cost(700, get_plan("api-pro")) == 0
    assert usage_cost(1000, get_plan("api-pro")) == 0


def test_second_tier_invoice():
    inv = build_invoice("acme", "api-pro", EVENTS, "2024-10")
    assert inv["lines"] == [("Base fee", 2000), ("Usage (3,000 calls)", 4000)]
    assert format_cents(inv["total_cents"]) == "$60.00"


def test_volume_pricing():
    plan = get_plan("api-volume")
    assert usage_cost(5000, plan) == 15000
    assert usage_cost(20000, plan) == 40000
    assert usage_cost(60000, plan) == 60000
