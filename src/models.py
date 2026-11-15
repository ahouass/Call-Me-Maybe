
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Describe a function parameter's type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "number",
        "string",
        "boolean",
        "integer",
        "array",
        "object"
    ]


class FunctionDefinition(BaseModel):
    """Describe a function available to the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class TestPrompt(BaseModel):
    """Represent one natural-language test request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
