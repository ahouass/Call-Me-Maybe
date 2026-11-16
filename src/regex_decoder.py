
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
        "rx_vowels": "Match each individual vowel character.",
        "rx_chars": "Match each character from a specified set.",
        "rx_digits": "Match every sequence of digits.",
        "rx_word": "Match one complete word.",
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
    """Ask Qwen to select a regex matching strategy."""
    prompt = (
        "Choose the correct regex matching strategy.\n"
        "Choose what the user wants to FIND and replace, "
        "not the replacement text.\n\n"
        "Available strategies:\n"
        "rx_vowels: match each vowel individually.\n"
        "rx_chars: match characters from a specified set.\n"
        "rx_digits: match digit sequences.\n"
        "rx_word: match one complete word.\n"
        "rx_literal: match an exact substring.\n\n"
        "Examples:\n"
        "Replace vowels in elephant -> rx_vowels\n"
        "Replace every digit in A12 -> rx_digits\n"
        "Replace the word bird with fox -> rx_word\n"
        "Replace each x or z -> rx_chars\n\n"
        f"Request: {request}\n"
        f"Source: {source}\n"
        f"{feedback}\n"
        "Select the matching strategy.\n"
        "JSON:\n"
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
    """Use Qwen to extract the text that should be matched."""
    if mode == "rx_word":
        words = list(
            dict.fromkeys(re.findall(r"\b\w+\b", source))
        )

        if not words:
            raise ValueError("No words found in source")

        candidates = [
            FunctionDefinition(
                name=word,
                description=f"Word appearing in source: {word}",
                parameters={},
                returns=ParameterDefinition(type="string")
            )
            for word in words
        ]

        word_prompt = (
            "Select the OLD word the user wants to replace.\n"
            "The OLD word must appear in the source text.\n"
            "Do NOT select the NEW replacement word.\n"
            "Example: Replace 'red' with 'blue' in 'red car'.\n"
            "Correct target: red\n\n"
            f"Request: {request}\n"
            f"Source: {source}\n"
            f"Available source words: {json.dumps(words)}\n"
            "Choose the word to FIND, not the replacement.\n"
            "JSON:\n"
        )

        return generate_name(
            model,
            word_prompt,
            candidates,
            tokens
        )

    prompt = (
        "Extract the target text for a Python regex.\n"
        "Return only a JSON string value.\n"
        "Extract what must be FOUND, not the replacement.\n"
        "For rx_chars, list the distinct target characters "
        "without brackets.\n"
        "Example: replace x and z -> xz.\n"
        "For rx_literal, return the exact substring to find.\n"
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
    """Construct a regex from an LLM-selected strategy."""
    if mode == "rx_vowels":
        return r"[aeiouAEIOU]"

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
    """Check whether a regex follows its selected strategy."""
    if mode == "rx_word":
        if re.fullmatch(r"\w+", target) is None:
            return False

    try:
        matches = list(re.finditer(pattern, source))
    except re.error:
        return False

    if not matches:
        return False

    if any(match.start() == match.end() for match in matches):
        return False

    if mode == "rx_chars":
        if any(len(match.group()) != 1 for match in matches):
            return False

    if mode == "rx_digits":
        if any(not match.group().isdigit() for match in matches):
            return False

    return True


def generate_regex(
    model: Small_LLM_Model,
    request: str,
    source: str,
    tokens: dict[int, str],
    max_attempts: int = 3
) -> str:
    """Generate a regex using LLM-selected strategies."""
    feedback = ""

    for attempt in range(1, max_attempts + 1):
        mode = choose_strategy(
            model,
            request,
            source,
            tokens,
            feedback
        )

        target = ""

        if mode not in ("rx_digits", "rx_vowels"):
            target = extract_target(
                model,
                request,
                source,
                mode,
                tokens
            )

        pattern = build_pattern(mode, target)

        print(
            f"Regex attempt {attempt}: "
            f"mode={mode}, "
            f"target={target!r}, "
            f"pattern={pattern!r}"
        )

        if is_usable(mode, target, pattern, source):
            return pattern

        feedback = (
            f"Previous strategy {mode} with target "
            f"{target!r} produced {pattern!r}, "
            "which was rejected. "
            "Choose a strategy and target that match "
            "the text the user wants to replace."
        )

    raise ValueError("No usable regex generated")
