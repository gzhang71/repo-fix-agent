from .keys import cache_key


class CachedClient:
    def __init__(self, fetch, cache):
        self._fetch = fetch
        self._cache = cache

    def get(self, endpoint, **params):
        key = cache_key(endpoint, params)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        value = self._fetch(endpoint, params)
        self._cache.set(key, value)
        return value
