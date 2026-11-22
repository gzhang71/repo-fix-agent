import pytest

from weather.display import format_reading
from weather.units import convert_temp, f_to_k


def test_f_to_k():
    assert f_to_k(32) == pytest.approx(273.15)


def test_celsius_paths():
    assert convert_temp(100, "C", "K") == pytest.approx(373.15)
    assert convert_temp(212, "F", "C") == pytest.approx(100)


def test_display_celsius():
    assert format_reading(293.15, "C") == "20°C"


def test_unknown_unit():
    with pytest.raises(ValueError):
        convert_temp(1, "C", "R")
