from app.wiring import App


def test_first_user_is_admin():
    app = App()
    assert app.signup("a@x.io")["role"] == "admin"
    assert app.signup("b@x.io")["role"] == "member"


def test_signup_is_audited():
    app = App()
    app.signup("a@x.io"); app.signup("b@x.io")
    assert [e for _, e in app.audit.entries] == ["a@x.io", "b@x.io"]
