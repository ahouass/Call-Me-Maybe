
"""Constrained decoding of JSON boolean values."""

from llm_sdk import Small_LLM_Model


def generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
    tokens: dict[int, str]
) -> str:
    """Generate true or false with token-level constraints."""
    choices = ("true", "false")
    generated = ""

    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = [
            token_id
            for token_id, token in tokens.items()
            if token_id < len(logits)
            and any(
                choice.startswith(generated + token)
                for choice in choices
            )
        ]

        if not allowed_ids:
            raise ValueError(
                "No valid boolean continuation"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        generated += tokens[next_id]
        input_ids.append(next_id)

        if generated in choices:
            return generated

    raise ValueError(
        "Boolean generation limit reached"
    )
