
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import FunctionDefinition, TestPrompt


def read_json(path: str) -> Any:
    """Read and parse a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as exc:
        raise ValueError(
            f"File not found: {path}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    except OSError as exc:
        raise ValueError(
            f"Cannot read {path}: {exc}"
        ) from exc


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions."""
    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError("Functions file must contain a list")

    try:
        return [
            FunctionDefinition.model_validate(item)
            for item in data
        ]
    except ValidationError as exc:
        raise ValueError(
            f"Invalid function definition: {exc}"
        ) from exc


def load_prompts(path: str) -> list[TestPrompt]:
    """Load and validate user prompts."""
    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError("Prompts file must contain a list")

    try:
        return [
            TestPrompt.model_validate(item)
            for item in data
        ]
    except ValidationError as exc:
        raise ValueError(
            f"Invalid test prompt: {exc}"
        ) from exc
