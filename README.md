*This project has been created as part of the 42 curriculum by ahouass.*

# Call Me Maybe

## Introduction

Call Me Maybe is a Python project that implements function calling using a small language model.

The application receives natural-language requests and a JSON file describing available functions. It uses **Qwen3-0.6B**, through the provided `llm_sdk`, to select the appropriate function and extract its arguments.

The results are saved in JSON format.

## Requirements

- Python 3.10 or newer
- uv for dependency management
- The provided `llm_sdk`
- Access to the Qwen3-0.6B model weights

## Installation

From the project root:

```bash
uv sync
```

Alternatively:

```bash
make sync
```

## Usage

Run the application with the default input files:

```bash
uv run python -m src
```

Or specify custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The default result file is:

`data/output/function_calling_results.json`

## Example

Input:

```text
What is the sum of 2 and 3?
```

Generated output:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

## Architecture

The implementation is located inside `src/`.

- `__main__.py` — Command-line entry point and output generation
- `loader.py` — Loads input JSON and validates function definitions
- `models.py` — Pydantic data models
- `prompt_builder.py` — Constructs prompts for the LLM
- `name_decoder.py` — Constrains function-name generation
- `number_decoder.py` — Handles numeric arguments
- `string_decoder.py` — Handles JSON-string arguments
- `regex_decoder.py` — Constructs regex patterns using model-selected strategies
- `json_parser.py` — Contains incremental numeric parsing utilities
- `validator.py` — Validates completed function calls

## Decoding approach

The application uses the language model's token logits and applies restrictions to the available next-token candidates before selection.

Function names are constrained to the provided function definitions. Arguments are generated according to their expected parameter types.

For common regex replacement operations, the model selects a matching strategy and extracts a target when necessary. Python constructs the resulting regular expression.

The final function call is validated against the available function definitions.

The current implementation uses staged constraints rather than a fully general incremental JSON grammar. Numeric generation also includes a source-literal early-stopping optimization, and regex generation supports a limited set of strategies.

## Testing

Run the code-quality checks:

```bash
make check
```

Or run them individually:

```bash
uv run flake8 src
uv run python -m mypy src
```

Execute the application:

```bash
make run
```

The supplied dataset contains 11 requests covering addition, greetings, string reversal, square roots, and regex substitutions.

## Performance

A complete execution of the 11 supplied requests finished in **224.79 seconds** in the development environment.

Actual execution time depends on hardware, model initialization, cached model files, and inference performance.

## Limitations

The current implementation has been verified against the supplied examples. Additional testing is needed to establish reliability on unseen prompts, arbitrary schemas, Unicode string content, and more complex regular expressions.