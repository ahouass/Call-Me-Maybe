
"""Test function-name prefix constraints."""

from .loader import load_functions
from .name_constraint import check_name_prefix


def main() -> None:
    """Test valid and invalid name prefixes."""
    functions = load_functions(
        "data/input/functions_definition.json"
    )

    examples = [
        '{',
        '{"name":"',
        '{"name":"fn_',
        '{"name":"fn_add',
        '{"name":"fn_add_numbers',
        '{"name":"fn_add_numbers"',
        '{"name":"fn_wrong',
        '{"name":"fn_greet"',
    ]

    for text in examples:
        status = check_name_prefix(text, functions)
        print(f"{text!r:35} -> {status.value}")


if __name__ == "__main__":
    main()
