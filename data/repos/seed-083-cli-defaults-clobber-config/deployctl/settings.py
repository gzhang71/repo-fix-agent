DEFAULTS = {"region": "local", "log_level": "info"}


def resolve(file_cfg, cli_opts):
    """Precedence: DEFAULTS < config file < command-line flags.

    Options the user did not pass on the command line are None and are ignored.
    """
    merged = dict(DEFAULTS)
    merged.update(file_cfg or {})
    merged.update({k: v for k, v in cli_opts.items() if v is not None})
    return merged
