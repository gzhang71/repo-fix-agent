def team_summary(db, team):
    members = db.table("users").query().where(team=team)
    newest = members.order_by("-id").first()
    return {
        "members": members.count(),
        "admins": members.where(role="admin").count(),
        "active": members.where(active=True).count(),
        "newest": newest["name"] if newest else None,
    }


def top_spenders(db, n=3, min_total=0):
    q = db.table("users").query().where(total_cents__gte=min_total)
    return [r["name"] for r in q.order_by("-total_cents").limit(n).all()]
