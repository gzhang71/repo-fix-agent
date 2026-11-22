from .querystring import get_first


def search_terms(url):
    """Lower-cased search terms from the `q` parameter of a request URL."""
    _, _, qs = url.partition("?")
    return get_first(qs, "q", "").lower().split()
