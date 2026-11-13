
"""Generate complete function calls with constrained decoding."""

import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .name_decoder import generate_name
from .number_decoder import generate_number
from .prompt_builder import build_prompt
from .string_decoder import (
    generate_json_string,
    load_text_tokens,
)
from .validator import validate_function_call


class FunctionDecoder:
    """Generate schema-constrained function calls using Qwen."""

    def __init__(self, model: Small_LLM_Model) -> None:
        """Initialize the decoder and token lookup."""
        self.model = model
        self.tokens = load_text_tokens(model)

    def generate(
        self,
        request: str,
        functions: list[FunctionDefinition]
    ) -> dict[str, Any]:
        """Generate a function call for a user request."""
        if not functions:
            raise ValueError("No functions available")

        prompt = build_prompt(request, functions)

        name = generate_name(
            self.model,
            prompt,
            functions,
            self.tokens
        )

        selected = next(
            item for item in functions
            if item.name == name
        )

        output = (
            '{"name":'
            + json.dumps(name)
            + ',"parameters":{'
        )

        parameters = list(selected.parameters.items())

        for index, (key, definition) in enumerate(parameters):
            last = index == len(parameters) - 1
            delimiter = "}" if last else ","

            output += json.dumps(key) + ":"

            if definition.type == "number":
                input_ids = self.model.encode(
                    prompt + output
                ).tolist()[0]

                value = generate_number(
                    self.model,
                    input_ids,
                    self.tokens,
                    delimiter
                )

                output += value + delimiter

            elif definition.type == "string":
                output += '"'

                input_ids = self.model.encode(
                    prompt + output
                ).tolist()[0]

                value = generate_json_string(
                    self.model,
                    input_ids,
                    self.tokens,
                    suffix="}}" if last else ","
                )

                output += value + delimiter

            else:
                raise ValueError(
                    f"Unsupported type: {definition.type}"
                )

        if not parameters:
            output += "}"

        output += "}"

        result: dict[str, Any] = json.loads(output)

        if not validate_function_call(result, functions):
            raise ValueError("Generated call failed validation")

        return result
