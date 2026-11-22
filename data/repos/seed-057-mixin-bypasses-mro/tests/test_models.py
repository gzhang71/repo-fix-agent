import pytest

from store.models import Customer, Note, Order


def test_customer_save_is_audited():
    store = {}
    Customer(store, id=1, name="Ann").save()
    assert store["customers"] == [{"id": 1, "name": "Ann"}]
    assert store["audit_log"] == [("customers", 1)]


def test_order_requires_customer():
    with pytest.raises(ValueError, match="customer_id"):
        Order({}, id=7).save()


def test_note_saved():
    store = {}
    Note(store, text="hi").save()
    assert store == {"notes": [{"text": "hi"}]}
