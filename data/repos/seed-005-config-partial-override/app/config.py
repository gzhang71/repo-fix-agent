DEFAULTS = {"timeout": 30, "retries": 3, "base_url": "https://api.example.com"}


def load_config(overrides=None):
    cfg = overrides or dict(DEFAULTS)
    return cfg
