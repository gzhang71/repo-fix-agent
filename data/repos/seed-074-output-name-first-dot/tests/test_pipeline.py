import pytest

from assets.naming import output_name
from assets.pipeline import plan_conversions


def test_output_name_simple():
    assert output_name("img/cat.png", ".webp") == "img/cat.webp"


def test_plan():
    assert plan_conversions(["a.png", "b.jpg"], ".webp") == {"a.png": "a.webp", "b.jpg": "b.webp"}


def test_plan_detects_collision():
    with pytest.raises(ValueError):
        plan_conversions(["a.png", "a.jpg"], ".webp")
