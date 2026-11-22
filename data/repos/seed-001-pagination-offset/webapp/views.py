from .pagination import paginate, total_pages


def list_view(items, page=1, per_page=10):
    return {"items": paginate(items, page, per_page),
            "pages": total_pages(len(items), per_page)}
