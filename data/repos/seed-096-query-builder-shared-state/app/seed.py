from minidb import Database

USERS = [
    ("ana", "engineering", "admin", True, 12000),
    ("ben", "engineering", "member", True, 4500),
    ("cy", "engineering", "admin", False, 800),
    ("dee", "engineering", "member", True, 15000),
    ("eve", "engineering", "member", False, 0),
    ("fay", "sales", "admin", True, 9900),
    ("gus", "engineering", "member", True, 3000),
]


def demo_db():
    db = Database()
    users = db.table("users")
    for name, team, role, active, total in USERS:
        users.insert(name=name, team=team, role=role, active=active, total_cents=total)
    return db
