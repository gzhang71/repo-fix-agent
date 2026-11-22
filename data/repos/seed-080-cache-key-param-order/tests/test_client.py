from fetchcache.client import CachedClient
from fetchcache.store import DiskCache


def make(tmp_path):
    calls = []

    def fetch(endpoint, params):
        calls.append((endpoint, dict(params)))
        return {"endpoint": endpoint, "n": len(calls)}

    return CachedClient(fetch, DiskCache(tmp_path / "cache")), calls


def test_second_identical_call_hits(tmp_path):
    client, calls = make(tmp_path)
    assert client.get("/a", x=1) == client.get("/a", x=1)
    assert len(calls) == 1


def test_different_params_miss(tmp_path):
    client, calls = make(tmp_path)
    client.get("/a", x=1)
    client.get("/a", x=2)
    assert len(calls) == 2
