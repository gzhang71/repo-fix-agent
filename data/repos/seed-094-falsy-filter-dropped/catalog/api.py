INT_FILTERS = {"min_price", "max_price", "min_rating"}


def parse_params(params):
    """Turn raw query-string values into typed filters and paging options."""
    filters, opts = {}, {}
    for key, raw in params.items():
        raw = raw.strip()
        if raw == "":
            continue
        if key in ("page", "per_page"):
            opts[key] = int(raw)
        elif key == "sort":
            opts["order_by"] = raw
        elif key == "in_stock":
            filters[key] = raw.lower() in ("1", "true", "yes")
        elif key in INT_FILTERS:
            filters[key] = int(raw)
        else:
            filters[key] = raw
    return filters, opts


def list_products(repo, params):
    filters, opts = parse_params(params)
    page = repo.search(filters, **opts)
    return {"total": page.total, "page": page.page, "ids": [p.id for p in page.items]}
