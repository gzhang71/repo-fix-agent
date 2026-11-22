from exporter.contacts import export_contacts


def test_simple_export():
    text = export_contacts([
        {"id": 2, "name": "Bob", "email": "bob@x.io", "notes": "vip"},
        {"id": 1, "name": "Ann", "email": "ann@x.io"},
    ])
    assert text == "id,name,email,notes\n1,Ann,ann@x.io,\n2,Bob,bob@x.io,vip\n"
