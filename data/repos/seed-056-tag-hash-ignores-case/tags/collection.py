from .model import Tag


def parse_tags(raw):
    """Parse a comma-separated tag string, keeping the first spelling of each tag."""
    seen = set()
    out = []
    for part in raw.split(","):
        if not part.strip():
            continue
        tag = Tag(part)
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def tag_counts(articles):
    """articles: iterable of lists of Tag. Returns {Tag: count}."""
    counts = {}
    for tags in articles:
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts
