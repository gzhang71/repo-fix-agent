from .template import InvoiceTemplate

MONTHLY = InvoiceTemplate(
    "monthly",
    lines=[{"desc": "Hosting", "amount": 100}],
    meta={"terms": "net30", "tags": ["recurring"]},
)


def invoice_for(customer, extras=()):
    inv = MONTHLY.clone(name=f"monthly-{customer}")
    inv.meta["customer"] = customer
    for desc, amount in extras:
        inv.lines.append({"desc": desc, "amount": amount})
    return inv
