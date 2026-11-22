from .catalog import get_plan
from .tiers import usage_cost
from .usage import total_units


def build_invoice(customer, plan_name, events, period):
    plan = get_plan(plan_name)
    units = total_units(events, customer, period)
    lines = [("Base fee", plan.base_cents),
             (f"Usage ({units:,} calls)", usage_cost(units, plan))]
    return {"customer": customer, "period": period, "lines": lines,
            "total_cents": sum(cents for _, cents in lines)}


def format_cents(cents):
    return f"${cents // 100:,}.{cents % 100:02d}"
