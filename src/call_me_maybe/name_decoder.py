
"""Constrained decoding of function names."""

import json

from llm_sdk import Small_LLM_Model

from .json_parser import ParseStatus
from .models import FunctionDefinition
from .name_constraint import check_name_prefix


NAME_PREFIX = '{"name":"'


def load_ascii_tokens(
    model: Small_LLM_Model
) -> dict[int, str]:
    """Load vocabulary tokens containing only ASCII characters."""
    with open(
        model.get_path_to_vocab_file(),
        "r",
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    return {
        token_id: raw
        for raw, token_id in vocab.items()
        if raw and raw.isascii()
        and model.decode([token_id]) == raw
    }


def generate_name(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition],
    ascii_tokens: dict[int, str]
) -> str:
    """Generate a valid function name using token masking."""
    input_ids = model.encode(prompt + NAME_PREFIX).tolist()[0]
    generated = ""

    for _ in range(30):
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids = [
            token_id
            for token_id, token_text in ascii_tokens.items()
            if token_id < len(logits)
            and check_name_prefix(
                NAME_PREFIX + generated + token_text,
                functions
            ) != ParseStatus.INVALID
        ]

        if not allowed_ids:
            raise ValueError("No valid continuation available")

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id]
        )

        generated += ascii_tokens[next_id]
        input_ids.append(next_id)

        status = check_name_prefix(
            NAME_PREFIX + generated,
            functions
        )

        if status == ParseStatus.COMPLETE:
            return generated[:-1]

    raise ValueError("Function-name generation limit reached")
