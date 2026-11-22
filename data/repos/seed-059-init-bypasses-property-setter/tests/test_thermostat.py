from climate.profiles import from_profile
from climate.thermostat import Thermostat


def test_setter_clamps():
    t = Thermostat()
    t.target = 95
    assert t.target == 30.0
    t.target = -4
    assert t.target == 10.0


def test_needs_heating():
    t = Thermostat(target=21)
    assert t.needs_heating(19.0)
    assert not t.needs_heating(20.8)


def test_profile_defaults():
    t = from_profile({})
    assert t.target == 20.0 and t.mode == "heat"
