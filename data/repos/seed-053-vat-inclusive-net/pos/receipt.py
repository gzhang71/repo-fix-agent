from decimal import Decimal

from .tax import net_from_gross, q


def vat_breakdown(gross, rate):
    gross = Decimal(str(gross))
    rate = Decimal(str(rate))
    net = net_from_gross(gross, rate)
    tax = q(gross * rate)
    return {"gross": gross, "net": net, "tax": tax}


def receipt_line(name, gross, rate):
    b = vat_breakdown(gross, rate)
    return f"{name}: net {b['net']}  VAT {b['tax']}  total {b['gross']}"
