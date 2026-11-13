
"""Constrained decoding of numeric JSON arguments."""

from llm_sdk import Small_LLM_Model

from .json_parser import ParseStatus, check_number_prefix


def check_number_continuation(
    text: str,
    delimiter: str
) -> ParseStatus:
    """Check a numeric prefix and its required delimiter."""
    if text.endswith(delimiter):
        number = text[:-len(delimiter)]

        if check_number_prefix(number) == ParseStatus.COMPLETE:
            return ParseStatus.COMPLETE

        return ParseStatus.INVALID

    if delimiter in text:
        return ParseStatus.INVALID

    if check_number_prefix(text) == ParseStatus.INVALID:
        return ParseStatus.INVALID

    return ParseStatus.INCOMPLETE


def generate_number(
    model: Small_LLM_Model,
    input_ids: list[int],
    ascii_tokens: dict[int, str],
    delimiter: str,
    max_number_length: int = 12
) -> str:
    """Generate a number with constrained token selection."""
    delimiter_id = next(
        (
            token_id
            for token_id, text in ascii_tokens.items()
            if text == delimiter
        ),
        None
    )

    if delimiter_id is None:
        raise ValueError("Delimiter token was not found")

    generated = ""

    for _ in range(40):
        # Force a valid delimiter once the length limit is reached.
        if len(generated) >= max_number_length:
            if check_number_prefix(generated) != ParseStatus.COMPLETE:
                raise ValueError("Number is incomplete at limit")

            input_ids.append(delimiter_id)
            print("Forced delimiter:", repr(delimiter))
            return generated

        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = []

        for token_id, token in ascii_tokens.items():
            if token_id >= len(logits):
                continue

            candidate = generated + token

            status = check_number_continuation(
                candidate,
                delimiter
            )

            if status == ParseStatus.INVALID:
                continue

            if status == ParseStatus.COMPLETE:
                number_part = candidate[:-len(delimiter)]
            else:
                number_part = candidate

            if len(number_part) > max_number_length:
                continue

            allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(
                f"No valid continuation after {generated!r}"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        token = ascii_tokens[next_id]
        generated += token
        input_ids.append(next_id)

        print("Generated:", repr(generated))

        if check_number_continuation(
            generated,
            delimiter
        ) == ParseStatus.COMPLETE:
            return generated[:-len(delimiter)]

    raise ValueError("Numeric generation limit reached")
