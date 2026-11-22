REGISTRY = {}


class UnknownFormat(KeyError):
    pass


class Exporter:
    """Base class for export plugins.

    Subclasses that declare a `format` are registered automatically so the API can
    look them up by name.
    """

    format = None
    content_type = "application/octet-stream"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.format is not None:
            REGISTRY[cls.format] = cls

    def export(self, rows):
        raise NotImplementedError


def get_exporter(fmt):
    try:
        return REGISTRY[fmt]()
    except KeyError:
        raise UnknownFormat(fmt) from None
