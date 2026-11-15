
"""Build prompts for the language model."""

from .models import FunctionDefinition




def build_prompt(
    user_request: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build a compact function-calling prompt."""
    descriptions = []

    for function in functions:
        args = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        )

        descriptions.append(
            f"{function.name}({args}): {function.description}"
        )

    available = "\n".join(descriptions)

    return (
        "Select the correct function and extract its arguments.\n"
        "Use only the listed functions and parameter names.\n"
        "Return JSON with name and parameters. No explanation.\n"
        "For regex, use Python syntax and match each occurrence "
        "individually. Avoid unnecessary .* patterns.\n\n"
        f"Functions:\n{available}\n\n"
        f"Request: {user_request}\n"
        "JSON:\n"
    )
