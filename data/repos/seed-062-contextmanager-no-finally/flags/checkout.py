from .registry import is_enabled


def checkout_url(cart_id):
    if is_enabled("new_checkout"):
        return f"/v2/checkout/{cart_id}"
    return f"/checkout/{cart_id}"
