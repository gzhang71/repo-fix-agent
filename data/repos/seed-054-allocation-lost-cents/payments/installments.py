from .allocate import split_evenly


def schedule(total, months):
    """[(month_number, amount), ...] for an interest-free installment plan."""
    return list(enumerate(split_evenly(total, months), start=1))
