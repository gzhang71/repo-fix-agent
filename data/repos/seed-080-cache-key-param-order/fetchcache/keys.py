import hashlib
import json


def cache_key(endpoint, params):
    payload = json.dumps({"endpoint": endpoint, "params": params})
    return hashlib.sha256(payload.encode()).hexdigest()
