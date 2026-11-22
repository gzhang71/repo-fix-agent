from .slug import slugify


def make_url(title, post_id):
    return f"/posts/{post_id}/{slugify(title)}"
