
"""Test completed function-call validation."""

from .loader import load_functions
from .validator import validate_function_call


def main() -> None:
    """Run function-call validation examples."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    cases = [
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3}
        },
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": "hello"}
        },
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2}
        },
        {
            "name": "fn_wrong",
            "parameters": {"a": 2, "b": 3}
        },
        {
            "name": "fn_greet",
            "parameters": {"name": "Alice"}
        },
        {
            "name": "fn_add_numbers",
            "parameters": {"a": True, "b": 3}
        }
    ]

    for index, case in enumerate(cases, start=1):
        result = validate_function_call(case, functions)
        print(f"Test {index}: {result}")


if __name__ == "__main__":
    main()
