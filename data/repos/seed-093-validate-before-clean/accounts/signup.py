from forms.cleaners import collapse_spaces, lower, strip
from forms.pipeline import Pipeline, ValidationError
from forms.rules import email, length, no_spaces, required


class SignupError(Exception):
    pass


SIGNUP = Pipeline(
    cleaners={
        "email": [strip, lower],
        "username": [strip],
        "display_name": [strip, collapse_spaces],
    },
    validators={
        "email": [required, email],
        "username": [required, length(3, 20), no_spaces],
        "display_name": [length(1, 40)],
    },
)


def signup(store, form):
    try:
        data = SIGNUP.run(form)
    except ValidationError as e:
        raise SignupError(e.errors) from None
    return store.create(data)
