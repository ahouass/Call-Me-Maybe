
"""Constrained decoding of JSON integer values."""

import re

from llm_sdk import Small_LLM_Model

from .json_parser import ParseStatus


INTEGER_PATTERN = r"-?(?:0|[1-9][0-9]*)"


def check_integer_continuation(
    text: str,
    delimiter: str
) -> ParseStatus:
    """Validate an integer prefix and its delimiter."""
    if text.endswith(delimiter):
        number = text[:-len(delimiter)]

        if re.fullmatch(INTEGER_PATTERN, number):
            return ParseStatus.COMPLETE

        return ParseStatus.INVALID

    if delimiter in text:
        return ParseStatus.INVALID

    if text == "-":
        return ParseStatus.INCOMPLETE

    if re.fullmatch(INTEGER_PATTERN, text):
        return ParseStatus.INCOMPLETE

    return ParseStatus.INVALID


def generate_integer(
    model: Small_LLM_Model,
    input_ids: list[int],
    tokens: dict[int, str],
    delimiter: str,
    max_characters: int = 24
) -> str:
    """Generate an integer with token-level constraints."""
    generated = ""

    for _ in range(40):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = []

        for token_id, token in tokens.items():
            if token_id >= len(logits):
                continue

            candidate = generated + token

            status = check_integer_continuation(
                candidate,
                delimiter
            )

            if status == ParseStatus.INVALID:
                continue

            if status == ParseStatus.COMPLETE:
                number = candidate[:-len(delimiter)]
            else:
                number = candidate

            if len(number) > max_characters:
                continue

            if (
                len(generated) >= max_characters
                and status != ParseStatus.COMPLETE
            ):
                continue

            allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(
                "No valid integer continuation"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        generated += tokens[next_id]
        input_ids.append(next_id)

        status = check_integer_continuation(
            generated,
            delimiter
        )

        if status == ParseStatus.COMPLETE:
            return generated[:-len(delimiter)]

    raise ValueError("Integer generation limit reached")
