from invoicing.template import InvoiceTemplate


def test_total():
    t = InvoiceTemplate("t", lines=[{"desc": "a", "amount": 3}, {"desc": "b", "amount": 4}])
    assert t.total() == 7


def test_clone_overrides_attributes():
    t = InvoiceTemplate("t", lines=[{"desc": "a", "amount": 3}])
    c = t.clone(name="copy")
    assert (c.name, t.name) == ("copy", "t")
    assert c.total() == 3
