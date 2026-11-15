
"""Generate regex patterns using LLM-selected strategies."""

import json
import re

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, ParameterDefinition
from .name_decoder import generate_name
from .string_decoder import generate_json_string


def regex_strategies() -> list[FunctionDefinition]:
    """Define available regex matching strategies."""
    descriptions = {
        "rx_chars": "Match each character from a set individually.",
        "rx_digits": "Match every sequence of digits.",
        "rx_word": "Match a complete word.",
        "rx_literal": "Match an exact literal substring."
    }

    return [
        FunctionDefinition(
            name=name,
            description=description,
            parameters={},
            returns=ParameterDefinition(type="string")
        )
        for name, description in descriptions.items()
    ]



def choose_strategy(
    model: Small_LLM_Model,
    request: str,
    source: str,
    tokens: dict[int, str],
    feedback: str
) -> str:
    """Ask Qwen to choose the appropriate regex strategy."""
    prompt = (
        "Choose a regex strategy for what the user wants "
        "to REPLACE, not for the source sentence.\n\n"
        "Strategies:\n"
        "rx_chars: individual characters from a set, "
        "such as each vowel.\n"
        "rx_digits: each sequence of digits.\n"
        "rx_word: one complete word, not a sentence.\n"
        "rx_literal: an exact literal substring.\n\n"
        "Examples:\n"
        "Request: Replace every vowel in 'education'\n"
        'Answer: {"name":"rx_chars"}\n'
        "Request: Replace numbers in 'A12 B34'\n"
        'Answer: {"name":"rx_digits"}\n'
        "Request: Replace the word 'bird' with 'fox'\n"
        'Answer: {"name":"rx_word"}\n'
        "Request: Replace 'hello' with 'goodbye'\n"
        'Answer: {"name":"rx_literal"}\n\n'
        f"Request: {request}\n"
        f"Source: {source}\n"
        f"{feedback}\n"
        "Answer: "
    )

    return generate_name(
        model,
        prompt,
        regex_strategies(),
        tokens
    )



def extract_target(
    model: Small_LLM_Model,
    request: str,
    source: str,
    mode: str,
    tokens: dict[int, str]
) -> str:
    """Ask Qwen to extract the target characters or text."""
    prompt = (
        "Extract the target text for a Python regex.\n"
        "Return only a JSON string value.\n"
        "For rx_chars, list the distinct target characters "
        "without brackets. Example: vowels -> aeiou.\n"
        "For rx_word, return only the target word.\n"
        "For rx_literal, return the exact target substring.\n"
        f"Strategy: {mode}\n"
        f"Request: {request}\n"
        f"Source: {source}\n"
        'Value: "'
    )

    input_ids = model.encode(prompt).tolist()[0]

    generated = generate_json_string(
        model,
        input_ids,
        tokens,
        suffix="",
        max_characters=50
    )

    return json.loads('"' + generated)


def build_pattern(mode: str, target: str) -> str:
    """Convert a selected strategy into Python regex syntax."""
    if mode == "rx_digits":
        return r"\d+"

    if not target:
        raise ValueError("Empty regex target")

    if mode == "rx_chars":
        characters = dict.fromkeys(target)
        escaped = "".join(
            re.escape(char) for char in characters
        )
        return "[" + escaped + "]"

    if mode == "rx_word":
        return r"\b" + re.escape(target) + r"\b"

    if mode == "rx_literal":
        return re.escape(target)

    raise ValueError(f"Unknown regex strategy: {mode}")



def is_usable(
    mode: str,
    target: str,
    pattern: str,
    source: str
) -> bool:
    """Check whether the regex matches its selected strategy."""
    if mode == "rx_word":
        if re.fullmatch(r"\w+", target) is None:
            return False

    try:
        matches = list(re.finditer(pattern, source))
    except re.error:
        return False

    if not matches:
        return False

    if any(m.start() == m.end() for m in matches):
        return False

    if mode == "rx_chars":
        if any(len(m.group()) != 1 for m in matches):
            return False

    if mode == "rx_digits":
        if any(not m.group().isdigit() for m in matches):
            return False

    return True



def generate_regex(
    model: Small_LLM_Model,
    request: str,
    source: str,
    tokens: dict[int, str],
    max_attempts: int = 3
) -> str:
    """Generate a regex using model-selected strategies."""
    feedback = ""

    for attempt in range(1, max_attempts + 1):
        mode = choose_strategy(
            model, request, source, tokens, feedback
        )

        target = ""

        if mode != "rx_digits":
            target = extract_target(
                model, request, source, mode, tokens
            )

        pattern = build_pattern(mode, target)

        print(
            f"Regex attempt {attempt}: "
            f"mode={mode}, pattern={pattern!r}"
        )

        if is_usable(mode, target, pattern, source):
            return pattern

        feedback = (
            f"Previous strategy {mode} produced "
            f"{pattern!r}, which did not match "
            "the source. Choose another strategy."
        )

    raise ValueError("No usable regex generated")
