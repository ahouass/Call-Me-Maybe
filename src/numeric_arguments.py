
"""Extract numeric literals and explicit parameter assignments."""

import re


NUMBER_PATTERN = (
    r"(?<![\w.])-?(?:0|[1-9]\d*)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?![\w]|\.\d)"
)


def extract_source_numbers(request: str) -> list[str]:
    """Extract numbers while allowing trailing punctuation."""
    return re.findall(NUMBER_PATTERN, request)


def extract_named_numbers(
    request: str,
    keys: list[str]
) -> dict[str, str]:
    """Find explicitly named numeric parameter assignments."""
    values: dict[str, str] = {}

    for key in keys:
        pattern = (
            rf"(?<!\w)(?:parameter\s+|argument\s+)?"
            rf"{re.escape(key)}"
            rf"\s*(?:=|:|(?:is\s+)?equal\s+to|equals|is)"
            rf"\s*({NUMBER_PATTERN})"
        )

        matches = re.findall(
            pattern,
            request,
            flags=re.IGNORECASE
        )

        if len(matches) == 1:
            values[key] = matches[0]

    return values
