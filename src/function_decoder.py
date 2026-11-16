
"""Generate complete function calls with constrained decoding."""

import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .name_decoder import generate_name
from .number_choice import (
    choose_source_number,
    has_prefix_collision,
)
from .number_decoder import generate_number
from .numeric_arguments import (
    extract_named_numbers,
    extract_source_numbers,
)
from .prompt_builder import build_prompt
from .regex_decoder import generate_regex
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

        # Extract numbers in their original order.
        # Keep duplicates such as "5 and 5".
        source_numbers = extract_source_numbers(request)

        # Let Qwen select the requested function.
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

        # Use a smaller prompt for argument generation.
        argument_prompt = build_prompt(request, [selected])

        output = (
            '{"name":'
            + json.dumps(name)
            + ',"parameters":{'
        )

        parameters = list(selected.parameters.items())

        # Identify numeric parameters in the selected function.
        numeric_keys = [
            key
            for key, definition in parameters
            if definition.type == "number"
        ]

        # Extract explicitly assigned numeric parameters.
        # Example: "parameter b equal to 2".
        named_numbers = extract_named_numbers(
            request,
            numeric_keys
        )

        parameter_values: dict[str, Any] = {}

        # Fast positional mapping is only used when no
        # explicit numeric assignments were found.
        ordered_numbers = (
            not named_numbers
            and len(parameters) > 0
            and len(source_numbers) == len(parameters)
            and all(
                definition.type == "number"
                for _, definition in parameters
            )
        )

        for index, (key, definition) in enumerate(parameters):
            last = index == len(parameters) - 1
            delimiter = "}" if last else ","

            output += json.dumps(key) + ":"

            if definition.type == "number":

                # Priority 1: Explicit parameter assignment.
                # Example: "parameter a equal to 265".
                if key in named_numbers:
                    value = named_numbers[key]

                    print(
                        f"Explicit parameter {key}: "
                        f"{value!r}"
                    )

                # Priority 2: Ordered numeric arguments.
                # Example: "What is the sum of 2 and 265?"
                elif ordered_numbers:
                    value = source_numbers[index]

                    print(
                        f"Mapped source number for {key}: "
                        f"{value!r}"
                    )

                # Priority 3: Resolve numeric prefix collisions.
                # Example: "2" and "265".
                elif has_prefix_collision(source_numbers):
                    value = choose_source_number(
                        self.model,
                        request,
                        key,
                        index,
                        source_numbers,
                        self.tokens
                    )

                    print(
                        f"Selected source number for {key}: "
                        f"{value!r}"
                    )

                # Priority 4: General constrained number decoding.
                else:
                    input_ids = self.model.encode(
                        argument_prompt + output
                    ).tolist()[0]

                    value = generate_number(
                        self.model,
                        input_ids,
                        self.tokens,
                        delimiter,
                        source_numbers=set(source_numbers)
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
                # Generate a regex for the source string.
                pattern = generate_regex(
                    self.model,
                    request,
                    parameter_values["source_string"],
                    self.tokens
                )

                output += json.dumps(pattern) + delimiter
                parameter_values[key] = pattern

            elif definition.type == "string":
                # Generate a JSON string with token constraints.
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

        # Close the parameters object if it has no fields.
        if not parameters:
            output += "}"

        # Close the outer function-call object.
        output += "}"

        result: dict[str, Any] = json.loads(output)

        # Validate the final result against the function schema.
        if not validate_function_call(result, functions):
            raise ValueError("Generated call failed validation")

        return result
