from .units import convert_temp


def format_reading(kelvin, unit="C"):
    return f"{round(convert_temp(kelvin, 'K', unit))}°{unit}"
