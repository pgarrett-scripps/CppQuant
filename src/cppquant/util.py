from typing import Any


def convert_best_type(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def process_value(value: str) -> Any:
    if value == 'NA':
        return None
    else:
        return convert_best_type(value)

