
"""Test complete function-call generation."""

import json

from llm_sdk import Small_LLM_Model

from .function_decoder import FunctionDecoder
from .loader import load_functions, load_prompts


def main() -> None:
    """Generate function calls for all input prompts."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    prompts = load_prompts(
        "data/input/function_calling_tests.json"
    )

    model = Small_LLM_Model()
    decoder = FunctionDecoder(model)

    for index, item in enumerate(prompts[8:], start=9):
        print(f"\nTEST {index}/{len(prompts)}")
        print("Prompt:", item.prompt, flush=True)

        try:
            result = decoder.generate(item.prompt, functions)
            print(json.dumps(result, indent=2))
        except (ValueError, TypeError) as exc:
            print("Generation error:", exc)


if __name__ == "__main__":
    main()
