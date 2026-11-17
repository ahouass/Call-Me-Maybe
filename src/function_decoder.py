
"""Generate complete function calls with constrained decoding."""

import json
import re
from typing import Any

from llm_sdk import Small_LLM_Model

from .boolean_decoder import generate_boolean
from .integer_decoder import generate_integer
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
    explicit_string_hint,
    generate_json_string,
    load_text_tokens,
)
from .validator import validate_function_call


INTEGER_PATTERN = r"-?(?:0|[1-9][0-9]*)"


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

        # Preserve number order and duplicate values.
        source_numbers = extract_source_numbers(request)

        # Select the requested function using Qwen.
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

        # Both number and integer parameters are numeric.
        numeric_keys = [
            key
            for key, definition in parameters
            if definition.type in ("number", "integer")
        ]

        # Extract assignments such as:
        # "parameter a equal to 265".
        named_numbers = extract_named_numbers(
            request,
            numeric_keys
        )

        parameter_values: dict[str, Any] = {}

        # Use ordered mapping only when all parameters
        # are numeric and source values match their types.
        ordered_numbers = (
            not named_numbers
            and len(parameters) > 0
            and len(source_numbers) == len(parameters)
            and all(
                definition.type in ("number", "integer")
                for _, definition in parameters
            )
            and all(
                definition.type != "integer"
                or re.fullmatch(
                    INTEGER_PATTERN,
                    source_numbers[index]
                ) is not None
                for index, (_, definition) in enumerate(parameters)
            )
        )

        for index, (key, definition) in enumerate(parameters):
            last = index == len(parameters) - 1
            delimiter = "}" if last else ","

            output += json.dumps(key) + ":"

            # -----------------------------------------
            # NUMBER AND INTEGER PARAMETERS
            # -----------------------------------------
            if definition.type in ("number", "integer"):

                # Integers cannot contain decimal points
                # or scientific notation.
                if definition.type == "integer":
                    eligible_numbers = [
                        number
                        for number in source_numbers
                        if re.fullmatch(
                            INTEGER_PATTERN,
                            number
                        ) is not None
                    ]
                else:
                    eligible_numbers = source_numbers

                # Priority 1: Explicit named assignment.
                if key in named_numbers:
                    value = named_numbers[key]

                    print(
                        f"Explicit parameter {key}: "
                        f"{value!r}"
                    )

                # Priority 2: Ordered source numbers.
                elif ordered_numbers:
                    value = source_numbers[index]

                    print(
                        f"Mapped source number for {key}: "
                        f"{value!r}"
                    )

                # Priority 3: Resolve numeric prefixes.
                elif has_prefix_collision(eligible_numbers):
                    value = choose_source_number(
                        self.model,
                        request,
                        key,
                        index,
                        eligible_numbers,
                        self.tokens
                    )

                    print(
                        f"Selected source number for {key}: "
                        f"{value!r}"
                    )

                # Priority 4: Token-constrained generation.
                else:
                    input_ids = self.model.encode(
                        argument_prompt + output
                    ).tolist()[0]

                    if definition.type == "integer":
                        value = generate_integer(
                            self.model,
                            input_ids,
                            self.tokens,
                            delimiter
                        )

                    else:
                        value = generate_number(
                            self.model,
                            input_ids,
                            self.tokens,
                            delimiter,
                            source_numbers=set(
                                eligible_numbers
                            )
                        )

                # Validate integer syntax even when
                # the value came from source extraction.
                if (
                    definition.type == "integer"
                    and re.fullmatch(
                        INTEGER_PATTERN,
                        value
                    ) is None
                ):
                    raise ValueError(
                        f"Invalid integer argument: {key}"
                    )

                output += value + delimiter
                parameter_values[key] = json.loads(value)

            # -----------------------------------------
            # REGEX STRING PARAMETERS
            # -----------------------------------------
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

            # -----------------------------------------
            # GENERAL STRING PARAMETERS
            # -----------------------------------------
            elif definition.type == "string":
                output += '"'

                input_ids = self.model.encode(
                    argument_prompt + output
                ).tolist()[0]

                value = generate_json_string(
                    self.model,
                    input_ids,
                    self.tokens,
                    suffix="}}" if last else ",",
                    literal_hint=explicit_string_hint(
                        request,
                        key
                    )
                )

                output += value + delimiter
                parameter_values[key] = json.loads('"' + value)

            # -----------------------------------------
            # BOOLEAN PARAMETERS
            # -----------------------------------------
            elif definition.type == "boolean":
                input_ids = self.model.encode(
                    argument_prompt + output
                ).tolist()[0]

                value = generate_boolean(
                    self.model,
                    input_ids,
                    self.tokens
                )

                output += value + delimiter
                parameter_values[key] = json.loads(value)

            # -----------------------------------------
            # UNSUPPORTED PARAMETER TYPES
            # -----------------------------------------
            else:
                raise ValueError(
                    f"Unsupported type: {definition.type}"
                )

        # Close empty parameter objects.
        if not parameters:
            output += "}"

        # Close the outer function-call object.
        output += "}"

        result: dict[str, Any] = json.loads(output)

        # Validate the result against its function schema.
        if not validate_function_call(result, functions):
            raise ValueError("Generated call failed validation")

        return result
