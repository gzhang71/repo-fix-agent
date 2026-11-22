class PositiveNumber:
    """Data descriptor for numeric attributes that must not be negative."""

    def __init__(self, default=0):
        self.default = default
        self.value = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value

    def __set__(self, obj, value):
        if value < 0:
            raise ValueError(f"{self.name} must be >= 0, got {value!r}")
        self.value = value
