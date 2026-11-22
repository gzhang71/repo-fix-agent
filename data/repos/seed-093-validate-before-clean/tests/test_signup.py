import pytest

from accounts.signup import SignupError, signup
from accounts.store import DuplicateUser, UserStore


def test_valid_signup():
    store = UserStore()
    user = signup(store, {"email": "bob@example.com", "username": "bob"})
    assert user["username"] == "bob"


def test_missing_email():
    with pytest.raises(SignupError) as e:
        signup(UserStore(), {"username": "bob"})
    assert e.value.args[0] == {"email": ["is required"]}


def test_short_username():
    with pytest.raises(SignupError) as e:
        signup(UserStore(), {"email": "a@b.io", "username": "bo"})
    assert e.value.args[0] == {"username": ["must be 3-20 characters"]}


def test_duplicate():
    store = UserStore()
    signup(store, {"email": "a@b.io", "username": "abc"})
    with pytest.raises(DuplicateUser):
        signup(store, {"email": "a@b.io", "username": "abd"})
