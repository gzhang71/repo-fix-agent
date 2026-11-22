class Bootstrap:
    """Makes the first user of a fresh install the workspace admin."""

    def __init__(self):
        self.admin = None

    def make_first_user_admin(self, user):
        user["role"] = "admin"
        self.admin = user["email"]
