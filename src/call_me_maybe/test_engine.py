
"""Test integration between the prompt builder and Qwen."""

from .llm_engine import LLMEngine
from .loader import load_functions
from .prompt_builder import build_prompt


def main() -> None:
    """Generate a function call for one request."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    prompt = build_prompt(
        "What is the sum of 2 and 3?",
        functions
    )

    engine = LLMEngine()

    result = engine.generate(prompt)

    print("Generated function call:")
    print(result)


if __name__ == "__main__":
    main()
