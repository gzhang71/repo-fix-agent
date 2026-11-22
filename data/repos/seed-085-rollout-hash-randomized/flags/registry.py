from .rollout import bucket


class FlagRegistry:
    """config: {flag: {"enabled": bool, "rollout": percent, "allow": [user ids]}}"""

    def __init__(self, config):
        self.config = config

    def is_enabled(self, flag, user_id):
        cfg = self.config.get(flag)
        if not cfg or not cfg.get("enabled", False):
            return False
        if user_id in cfg.get("allow", ()):
            return True
        return bucket(flag, user_id) < cfg.get("rollout", 0)
