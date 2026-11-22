from accounts.user import User


def test_display_name():
    assert User(1, "Ada", "Lovelace", "ada@x.io").display_name == "Ada Lovelace"


def test_display_name_without_last():
    assert User(2, "Cher", "", "cher@x.io").display_name == "Cher"
