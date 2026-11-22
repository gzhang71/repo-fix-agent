import logging

FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure(level="INFO", stream=None):
    """Configure the 'applog' logger. Safe to call again, e.g. on config reload."""
    logger = logging.getLogger("applog")
    handler = logging.StreamHandler(stream)
    handler.setLevel(level.upper())
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
