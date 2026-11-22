import pytest

from app.documents import Document
from app.setup import build_app
from authz.policy import PermissionDenied
from authz.roles import permissions_for


def test_role_inheritance():
    assert permissions_for("admin") == {"read", "write", "delete", "manage_members"}
    assert permissions_for("viewer") == {"read"}


def test_viewer_cannot_delete():
    members, docs = build_app()
    members.set_role("bob", "finance", "viewer")
    docs.add(Document(1, "finance", "Q3"))
    assert docs.open("bob", 1).title == "Q3"
    with pytest.raises(PermissionDenied):
        docs.delete("bob", 1)


def test_promotion_takes_effect():
    members, docs = build_app()
    members.set_role("bob", "finance", "viewer")
    docs.add(Document(1, "finance", "Q3"))
    docs.open("bob", 1)
    members.set_role("bob", "finance", "admin")
    docs.delete("bob", 1)
    assert 1 not in docs.docs


def test_cache_hits():
    members, docs = build_app()
    members.set_role("bob", "finance", "viewer")
    docs.add(Document(1, "finance", "Q3"))
    docs.open("bob", 1); docs.open("bob", 1)
    assert docs.authz.cache.hits == 1
