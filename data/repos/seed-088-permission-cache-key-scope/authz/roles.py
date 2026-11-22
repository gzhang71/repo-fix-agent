BASE_PERMISSIONS = {
    "viewer": {"read"},
    "editor": {"write"},
    "admin": {"delete", "manage_members"},
}
INHERITS = {"viewer": None, "editor": "viewer", "admin": "editor"}


def permissions_for(role):
    perms = set()
    while role is not None:
        perms |= BASE_PERMISSIONS[role]
        role = INHERITS[role]
    return frozenset(perms)
