
"""Generate complete function calls with constrained decoding."""

import json
import re
from typing import Any
from .regex_decoder import generate_regex

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

        source_numbers = set(
            re.findall(
                r"(?<![\w.])-?(?:0|[1-9]\d*)"
                r"(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])",
                request
            )
        )

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

        # Only the selected function is needed for its arguments.
        argument_prompt = build_prompt(request, [selected])

        output = (
            '{"name":'
            + json.dumps(name)
            + ',"parameters":{'
        )

        parameters = list(selected.parameters.items())

        # Store generated parameter values
        parameter_values: dict[str, Any] = {}

        for index, (key, definition) in enumerate(parameters):
            last = index == len(parameters) - 1
            delimiter = "}" if last else ","

            output += json.dumps(key) + ":"

            if definition.type == "number":
                input_ids = self.model.encode(
                    argument_prompt + output
                ).tolist()[0]

                value = generate_number(
                    self.model,
                    input_ids,
                    self.tokens,
                    delimiter,
                    source_numbers=source_numbers
                )

                output += value + delimiter
                parameter_values[key] = json.loads(value)

            elif (
                definition.type == "string"
                and key == "regex"
                and isinstance(
                    parameter_values.get("source_string"),
                    str
                )
            ):
                pattern = generate_regex(
                    self.model,
                    request,
                    parameter_values["source_string"],
                    self.tokens
                )

                output += json.dumps(pattern) + delimiter
                parameter_values[key] = pattern

            elif definition.type == "string":
                output += '"'

                input_ids = self.model.encode(
                    argument_prompt + output
                ).tolist()[0]

                value = generate_json_string(
                    self.model,
                    input_ids,
                    self.tokens,
                    suffix="}}" if last else ","
                )

                output += value + delimiter
                parameter_values[key] = json.loads('"' + value)

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
