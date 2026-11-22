MIN_TEMP = 10.0
MAX_TEMP = 30.0


class Thermostat:
    def __init__(self, target=20.0, mode="heat"):
        self._target = float(target)
        self.mode = mode

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        """Set the target temperature, clamped to the supported range."""
        self._target = min(MAX_TEMP, max(MIN_TEMP, float(value)))

    def needs_heating(self, current):
        return self.mode == "heat" and current < self.target - 0.5
