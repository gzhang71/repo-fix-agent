from forms.schema import validate
from forms.validators import max_length, required

GOOD = {"username": "ada", "email": "ada@x.io", "display_name": "Ada"}


def test_valid_data():
    assert validate(GOOD) == {}


def test_missing_display_name():
    assert validate({**GOOD, "display_name": ""}) == {"display_name": "display_name is required"}


def test_validator_functions():
    assert required("x", None) == "x is required"
    assert max_length(2)("x", "abc") == "x must be at most 2 characters"
