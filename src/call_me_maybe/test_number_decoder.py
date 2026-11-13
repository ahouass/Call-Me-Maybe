
"""Test numeric constrained decoding."""

from llm_sdk import Small_LLM_Model

from .loader import load_functions
from .name_decoder import load_ascii_tokens
from .number_decoder import generate_number
from .prompt_builder import build_prompt


def main() -> None:
    """Generate the first argument of an addition request."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    request = "What is the sum of 2 and 3?"
    prompt = build_prompt(request, functions)

    model = Small_LLM_Model()
    ascii_tokens = load_ascii_tokens(model)

    prefix = (
        '{"name":"fn_add_numbers",'
        '"parameters":{"a":'
    )

    input_ids = model.encode(
        prompt + prefix
    ).tolist()[0]

    number = generate_number(
        model,
        input_ids,
        ascii_tokens,
        ","
    )

    print("User request:", request)
    print("Generated argument a:", number)


if __name__ == "__main__":
    main()
