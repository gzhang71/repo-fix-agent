from app.reports import team_summary, top_spenders
from app.seed import demo_db


def test_where_and_order():
    q = demo_db().table("users").query().where(team="engineering", active=True)
    assert [r["name"] for r in q.order_by("name").all()] == ["ana", "ben", "dee", "gus"]


def test_lookups_and_limit():
    q = demo_db().table("users").query().where(total_cents__gt=5000).order_by("-total_cents").limit(2)
    assert [r["name"] for r in q.all()] == ["dee", "ana"]


def test_top_spenders():
    assert top_spenders(demo_db()) == ["dee", "ana", "fay"]


def test_sales_summary():
    assert team_summary(demo_db(), "sales") == {
        "members": 1, "admins": 1, "active": 1, "newest": "fay"}
