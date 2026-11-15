
"""Validate function calls against their definitions."""

import math
from typing import Any

from .models import FunctionDefinition


def valid_type(value: Any, expected: str) -> bool:
    """Check a value against a supported JSON parameter type."""
    if expected == "string":
        return isinstance(value, str)

    if expected == "boolean":
        return type(value) is bool

    if expected == "number":
        return (
            type(value) in (int, float)
            and math.isfinite(value)
        )

    if expected == "integer":
        return type(value) is int

    if expected == "array":
        return isinstance(value, list)

    if expected == "object":
        return isinstance(value, dict)

    return False


def validate_function_call(
    data: dict[str, Any],
    functions: list[FunctionDefinition]
) -> bool:
    """Validate a completed function call."""
    if set(data) != {"name", "parameters"}:
        return False

    name = data["name"]
    parameters = data["parameters"]

    if not isinstance(name, str):
        return False

    if not isinstance(parameters, dict):
        return False

    function = next(
        (item for item in functions if item.name == name),
        None
    )

    if function is None:
        return False

    if set(parameters) != set(function.parameters):
        return False

    for key, definition in function.parameters.items():
        if not valid_type(parameters[key], definition.type):
            return False

    return True
