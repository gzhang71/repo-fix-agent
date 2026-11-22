from promo.coupons import lookup
from promo.stacking import resolve


def price(cart):
    applied = resolve([lookup(code) for code in cart.codes])
    subtotal = cart.subtotal_cents
    discount = subtotal * sum(c.percent for c in applied) // 100
    discount += sum(c.amount_cents for c in applied)
    discount = min(discount, subtotal)
    return {"subtotal": subtotal, "applied": [c.code for c in applied],
            "discount": discount, "total": subtotal - discount}
