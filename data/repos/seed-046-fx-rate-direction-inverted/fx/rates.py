from decimal import Decimal

# Value of ONE unit of each currency, expressed in USD.
USD_PER_UNIT = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.25"),
    "JPY": Decimal("0.0067"),
}


def rate(code):
    try:
        return USD_PER_UNIT[code]
    except KeyError:
        raise ValueError(f"unknown currency {code!r}") from None
