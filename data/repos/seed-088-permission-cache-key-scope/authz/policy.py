from .roles import permissions_for


class PermissionDenied(Exception):
    pass


class Authorizer:
    def __init__(self, memberships, cache):
        self.memberships = memberships
        self.cache = cache

    def _compute(self, user_id, project_id):
        role = self.memberships.role_of(user_id, project_id)
        return permissions_for(role) if role else frozenset()

    def can(self, user_id, action, project_id):
        perms = self.cache.get_or_compute(
            user_id, project_id, lambda: self._compute(user_id, project_id))
        return action in perms

    def require(self, user_id, action, project_id):
        if not self.can(user_id, action, project_id):
            raise PermissionDenied(f"{user_id} may not {action} in {project_id}")
