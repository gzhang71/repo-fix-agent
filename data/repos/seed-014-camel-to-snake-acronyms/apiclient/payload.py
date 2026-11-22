from .casing import camel_to_snake


def normalize_keys(obj):
    """Recursively convert camelCase dict keys from the API to snake_case."""
    if isinstance(obj, dict):
        return {camel_to_snake(k): normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(v) for v in obj]
    return obj
