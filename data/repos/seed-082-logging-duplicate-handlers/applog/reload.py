from .setup import configure


def apply_config(cfg, stream=None):
    """Called at startup and again whenever the config file changes."""
    return configure(cfg.get("log_level", "INFO"), stream)
