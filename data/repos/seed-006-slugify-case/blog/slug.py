import re


def slugify(title):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title)
    return slug.strip("-")
