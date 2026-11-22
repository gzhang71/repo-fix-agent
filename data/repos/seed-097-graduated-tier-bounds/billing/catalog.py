from .models import Plan, Tier

PLANS = {
    "api-pro": Plan("api-pro", 2000, (Tier(1000, 0), Tier(10000, 2), Tier(None, 1)), "graduated"),
    "api-volume": Plan("api-volume", 0, (Tier(5000, 3), Tier(50000, 2), Tier(None, 1)), "volume"),
}


def get_plan(name):
    try:
        return PLANS[name]
    except KeyError:
        raise ValueError(f"unknown plan {name!r}") from None
