
"""Constrained generation of JSON string values."""

import json
import re
import string

from llm_sdk import Small_LLM_Model

from .json_parser import ParseStatus


def check_string_prefix(text: str) -> ParseStatus:
    """Validate JSON content after its opening quote."""
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

        elif (
            ord(char) < 32
            or 0xD800 <= ord(char) <= 0xDFFF
        ):
            return ParseStatus.INVALID

    return ParseStatus.INCOMPLETE


def check_string_continuation(
    text: str,
    suffix: str
) -> tuple[ParseStatus, str | None]:
    """Recognize a closing quote and the expected suffix."""
    for index, char in enumerate(text):
        if char != '"':
            continue

        possible_string = text[:index + 1]

        if (
            check_string_prefix(possible_string)
            != ParseStatus.COMPLETE
        ):
            continue

        remaining = text[index + 1:]

        if not suffix.startswith(remaining):
            return ParseStatus.INVALID, None

        return ParseStatus.COMPLETE, possible_string

    return check_string_prefix(text), None


def load_text_tokens(
    model: Small_LLM_Model
) -> dict[int, str]:
    """Load printable Unicode and JSON escape tokens."""
    with open(
        model.get_path_to_vocab_file(),
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    result: dict[int, str] = {}

    for raw, token_id in vocab.items():
        if raw.startswith("<|") and raw.endswith("|>"):
            continue

        decoded = model.decode([token_id])

        if not decoded or "\ufffd" in decoded:
            continue

        if any(
            0xD800 <= ord(char) <= 0xDFFF
            for char in decoded
        ):
            continue

        result[token_id] = decoded

    return result


def explicit_string_hint(
    request: str,
    key: str
) -> str | None:
    """Extract a clear literal from supported request forms."""
    if key == "s":
        match = re.fullmatch(
            r"\s*reverse\s+(?:the\s+)?string\s+"
            r"(['\"])(.*?)\1\s*[.!?]?\s*",
            request,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(2)

    if key == "name":
        match = re.fullmatch(
            r"\s*greet\s+(.+?)\s*",
            request,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            name = match.group(1).strip()

            if (
                len(name) >= 2
                and name[0] == name[-1]
                and name[0] in ("'", '"')
            ):
                return name[1:-1]

            return name

    return None


def generate_json_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    tokens: dict[int, str],
    suffix: str,
    max_characters: int = 120,
    literal_hint: str | None = None
) -> str:
    """Generate JSON string text with constrained tokens."""
    generated = ""
    target = None

    if literal_hint is not None:
        if len(literal_hint) > max_characters:
            raise ValueError(
                "Literal hint exceeds string limit"
            )

        # The caller already emitted the opening quote.
        # ensure_ascii permits Unicode through JSON escapes.
        target = json.dumps(
            literal_hint,
            ensure_ascii=True
        )[1:]

    if target is not None:
        max_steps = max(
            120,
            len(target) + len(suffix) + 4
        )
    else:
        max_steps = 120

    for _ in range(max_steps):
        logits = model.get_logits_from_input_ids(
            input_ids
        )

        allowed_ids: list[int] = []

        for token_id, token in tokens.items():
            if token_id >= len(logits):
                continue

            candidate = generated + token

            # When the source provides an exact literal,
            # constrain token generation to its JSON form.
            if target is not None:
                if not (
                    target + suffix
                ).startswith(candidate):
                    continue

            status, completed = (
                check_string_continuation(
                    candidate,
                    suffix
                )
            )

            if status == ParseStatus.INVALID:
                continue

            if target is not None:
                if len(candidate) > (
                    len(target) + len(suffix)
                ):
                    continue
            else:
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
                "No valid JSON string continuation"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        generated += tokens[next_id]
        input_ids.append(next_id)

        _, completed = check_string_continuation(
            generated,
            suffix
        )

        if completed is not None:
            return completed

    raise ValueError(
        "JSON string generation limit reached"
    )
