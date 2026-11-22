from .coupons import discount_for
from .models import Order
from .shipping import shipping_for
from .tax import compute_tax


def checkout(lines, region, coupon_code=None):
    order = Order(lines=list(lines), region=region, coupon_code=coupon_code)
    if coupon_code:
        order.discount_cents = discount_for(coupon_code, order.subtotal_cents)
    order.shipping_cents = shipping_for(order)
    order.tax_cents = compute_tax(order)
    return order


def receipt(order):
    return {
        "subtotal": order.subtotal_cents,
        "discount": order.discount_cents,
        "shipping": order.shipping_cents,
        "tax": order.tax_cents,
        "total": order.total_cents,
    }
