from .config import load_config


class Client:
    def __init__(self, overrides=None):
        self.cfg = load_config(overrides)

    def fetch(self, path):
        url = f"{self.cfg['base_url']}/{path}"
        return {"url": url, "timeout": self.cfg["timeout"], "retries": self.cfg["retries"]}
