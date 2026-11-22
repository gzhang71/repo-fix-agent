class MembershipStore:
    def __init__(self, cache=None):
        self._roles = {}
        self._cache = cache

    def set_role(self, user_id, project_id, role):
        self._roles[(user_id, project_id)] = role
        if self._cache is not None:
            self._cache.invalidate_user(user_id)

    def role_of(self, user_id, project_id):
        return self._roles.get((user_id, project_id))
