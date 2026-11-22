class PermissionCache:
    """Memoizes a user's effective permission set within a project."""

    def __init__(self):
        self._data = {}
        self.hits = 0

    def get_or_compute(self, user_id, project_id, compute):
        key = user_id
        if key in self._data:
            self.hits += 1
            return self._data[key]
        perms = compute()
        self._data[key] = perms
        return perms

    def invalidate_user(self, user_id):
        self._data.pop(user_id, None)
