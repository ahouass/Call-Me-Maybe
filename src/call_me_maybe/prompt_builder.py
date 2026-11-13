
"""Build prompts for the language model."""

import json

from .models import FunctionDefinition



def build_prompt(
    user_request: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build a prompt for accurate function calling."""
    definitions = [
        function.model_dump()
        for function in functions
    ]

    functions_json = json.dumps(definitions, indent=2)

    return (
        "You are a precise function-calling assistant.\n"
        "Select exactly one function from the definitions.\n"
        "Extract the arguments directly from the user's request.\n"
        "Do not execute the function.\n\n"
        f"Available functions:\n{functions_json}\n\n"
        "Argument extraction rules:\n"
        "1. Preserve strings exactly as requested.\n"
        "2. Extract numbers without unnecessary decimal digits.\n"
        "3. For regex replacement, match EACH target occurrence "
        "individually, not the entire input sentence.\n"
        "4. Use the shortest precise regex that performs "
        "the requested replacement.\n"
        "5. Avoid .* and repeated regex groups unless "
        "the request specifically requires them.\n"
        "6. For digit sequences, a typical pattern is \\d+.\n"
        "7. For lowercase vowels, a typical pattern "
        "is [aeiou].\n"
        "8. For complete words, word boundaries may help.\n"
        "9. Escape backslashes correctly in JSON strings.\n"
        "10. Never add extra text to argument values.\n\n"
        f"User request:\n{user_request}\n\n"
        "Return only a JSON object with this structure:\n"
        '{"name":"function_name","parameters":{}}\n'
        "JSON:\n"
    )
