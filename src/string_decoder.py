
"""Constrained generation of JSON string values."""

import json
import string

from llm_sdk import Small_LLM_Model

from .json_parser import ParseStatus


def check_string_prefix(text: str) -> ParseStatus:
    """Check JSON string content, including its closing quote."""
    escaped = False
    unicode_left = 0

    for index, char in enumerate(text):
        if unicode_left:
            if char not in string.hexdigits:
                return ParseStatus.INVALID
            unicode_left -= 1

        elif escaped:
            if char == "u":
                unicode_left = 4
            elif char not in '"\\/bfnrt':
                return ParseStatus.INVALID
            escaped = False

        elif char == "\\":
            escaped = True

        elif char == '"':
            if index == len(text) - 1:
                return ParseStatus.COMPLETE
            return ParseStatus.INVALID

        elif ord(char) < 32:
            return ParseStatus.INVALID

    return ParseStatus.INCOMPLETE


def load_text_tokens(
    model: Small_LLM_Model
) -> dict[int, str]:
    """Build a reusable map of printable ASCII token text."""
    with open(
        model.get_path_to_vocab_file(),
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    result = {}

    for raw, token_id in vocab.items():
        if not raw.replace("Ġ", " ").isascii():
            continue

        decoded = model.decode([token_id])

        if decoded and all(32 <= ord(c) <= 126 for c in decoded):
            result[token_id] = decoded

    return result



def generate_json_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    tokens: dict[int, str],
    suffix: str,
    max_characters: int = 120
) -> str:
    """Generate a JSON string with awareness of its suffix."""
    generated = ""

    for _ in range(120):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = []

        for token_id, token in tokens.items():
            if token_id >= len(logits):
                continue

            candidate = generated + token

            status, completed = check_string_continuation(
                candidate,
                suffix
            )

            if status == ParseStatus.INVALID:
                continue

            if len(candidate) > (
                max_characters + 1 + len(suffix)
            ):
                continue

            if (
                len(generated) >= max_characters
                and completed is None
            ):
                continue

            allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(
                "No valid string continuation"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        generated += tokens[next_id]
        input_ids.append(next_id)

        status, completed = check_string_continuation(
            generated,
            suffix
        )

        if completed is not None:
            return completed

    raise ValueError("String generation limit reached")




def check_string_continuation(
    text: str,
    suffix: str
) -> tuple[ParseStatus, str | None]:
    """Validate a string and any following JSON punctuation."""

    for index, char in enumerate(text):
        if char != '"':
            continue

        possible_string = text[:index + 1]

        if check_string_prefix(
            possible_string
        ) != ParseStatus.COMPLETE:
            continue

        remaining = text[index + 1:]

        if not suffix.startswith(remaining):
            return ParseStatus.INVALID, None

        return ParseStatus.COMPLETE, possible_string

    return check_string_prefix(text), None
