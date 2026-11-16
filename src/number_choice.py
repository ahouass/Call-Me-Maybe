
"""Constrained selection of numeric values from a request."""

import json

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, ParameterDefinition
from .name_decoder import generate_name


def has_prefix_collision(numbers: list[str]) -> bool:
    """Detect numeric strings that are prefixes of others."""
    unique = list(dict.fromkeys(numbers))

    return any(
        first != second and second.startswith(first)
        for first in unique
        for second in unique
    )


def choose_source_number(
    model: Small_LLM_Model,
    request: str,
    parameter_name: str,
    parameter_index: int,
    source_numbers: list[str],
    tokens: dict[int, str]
) -> str:
    """Ask the LLM to choose a complete numeric source value."""
    numbers = list(dict.fromkeys(source_numbers))

    if not numbers:
        raise ValueError("No source numbers available")

    if len(numbers) == 1:
        return numbers[0]

    choices = [
        FunctionDefinition(
            name=f"pick_{index}",
            description=f"Select the complete number {number}",
            parameters={},
            returns=ParameterDefinition(type="string")
        )
        for index, number in enumerate(numbers)
    ]

    prompt = (
        "Select the numeric value for a function argument.\n"
        "Choose a COMPLETE number from the candidates.\n"
        "Never select only the prefix of a larger number "
        "unless that shorter number is intended.\n"
        "For addition, argument a is the first number and "
        "argument b is the second number.\n\n"
        f"User request: {request}\n"
        f"Parameter: {parameter_name}\n"
        f"Parameter position: {parameter_index + 1}\n"
        f"Candidates: {json.dumps(numbers)}\n\n"
        "Available choices:\n"
    )

    for index, number in enumerate(numbers):
        prompt += f"pick_{index}: {number}\n"

    prompt += "\nChoose the correct candidate.\nJSON:\n"

    selected = generate_name(
        model,
        prompt,
        choices,
        tokens
    )

    for index, number in enumerate(numbers):
        if selected == f"pick_{index}":
            return number

    raise ValueError(f"Invalid numeric choice: {selected}")
