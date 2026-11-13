
"""Test incremental JSON number validation."""

from .json_parser import check_number_prefix


def main() -> None:
    """Display parsing results for sample values."""
    examples = [
        "",
        "-",
        "12",
        "-5",
        "3.14",
        "12.",
        "1e",
        "1e3",
        "12hello",
        "012",
        "true",
    ]

    for example in examples:
        status = check_number_prefix(example)

        print(f"{example!r:12} -> {status.value}")


if __name__ == "__main__":
    main()
