
"""Validate prefixes of JSON number values."""

import re
from enum import Enum


class ParseStatus(str, Enum):
    """Possible states of incremental parsing."""

    INCOMPLETE = "INCOMPLETE"
    COMPLETE = "COMPLETE"
    INVALID = "INVALID"


NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?"
)


def check_number_prefix(text: str) -> ParseStatus:
    """Check a JSON number prefix without surrounding JSON syntax."""
    if text in ("", "-", ".", "-."):
        if text in (".", "-."):
            return ParseStatus.INVALID
        return ParseStatus.INCOMPLETE

    match = NUMBER.fullmatch(text)

    if match is not None:
        return ParseStatus.COMPLETE

    partial = re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)\.",
        text
    )

    if partial is not None:
        return ParseStatus.INCOMPLETE

    partial = re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?",
        text
    )

    if partial is not None:
        return ParseStatus.INCOMPLETE

    return ParseStatus.INVALID
