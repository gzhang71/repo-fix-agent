def dedupe(items, key=None):
    """Yield items in order, skipping any whose key (default: the item) was already seen."""
    seen = set()
    for item in items:
        k = item if key is None else key(item)
        if k in seen:
            continue
        seen.add(item)
        yield item


def chunked(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
