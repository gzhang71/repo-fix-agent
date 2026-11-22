from .thermostat import Thermostat


def from_profile(profile):
    """Build a thermostat from a saved profile dict."""
    return Thermostat(target=profile.get("target", 20.0), mode=profile.get("mode", "heat"))
