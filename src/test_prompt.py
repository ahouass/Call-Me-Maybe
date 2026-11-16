
from .loader import load_functions
from .prompt_builder import build_prompt


functions = load_functions(
    "data/input/functions_definition.json"
)

prompt = build_prompt(
    "What is the sum of 2 and 3?",
    functions
)

print(prompt)
