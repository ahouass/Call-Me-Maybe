
"""Check incremental function-name generation."""

from .json_parser import ParseStatus
from .models import FunctionDefinition


NAME_PREFIX = '{"name":"'


def check_name_prefix(
    text: str,
    functions: list[FunctionDefinition]
) -> ParseStatus:
    """Check a partial function-name JSON field."""
    if NAME_PREFIX.startswith(text):
        return ParseStatus.INCOMPLETE

    if not text.startswith(NAME_PREFIX):
        return ParseStatus.INVALID

    remaining = text[len(NAME_PREFIX):]
    names = [function.name for function in functions]

    if remaining.endswith('"'):
        selected = remaining[:-1]
        if selected in names:
            return ParseStatus.COMPLETE
        return ParseStatus.INVALID

    if any(name.startswith(remaining) for name in names):
        return ParseStatus.INCOMPLETE

    return ParseStatus.INVALID
