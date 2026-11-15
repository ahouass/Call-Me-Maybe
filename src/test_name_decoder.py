
"""Test constrained function-name generation with Qwen."""

from llm_sdk import Small_LLM_Model

from .loader import load_functions
from .name_decoder import generate_name, load_ascii_tokens
from .prompt_builder import build_prompt


def main() -> None:
    """Test function selection from a natural-language request."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    model = Small_LLM_Model()
    ascii_tokens = load_ascii_tokens(model)

    request = "What is the sum of 2 and 3?"
    prompt = build_prompt(request, functions)

    name = generate_name(
        model,
        prompt,
        functions,
        ascii_tokens
    )

    print("User request:", request)
    print("Selected function:", name)


if __name__ == "__main__":
    main()
