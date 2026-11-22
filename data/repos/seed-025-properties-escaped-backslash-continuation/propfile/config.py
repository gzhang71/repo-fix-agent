from .reader import load


class Config:
    def __init__(self, text):
        self._props = load(text)

    def get(self, key, default=None):
        return self._props.get(key, default)

    def require(self, key):
        try:
            return self._props[key]
        except KeyError:
            raise KeyError(f"missing required property {key!r}") from None

    def get_list(self, key, sep=","):
        value = self.get(key, "")
        return [item.strip() for item in value.split(sep) if item.strip()]
