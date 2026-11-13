
import argparse
import sys

from .loader import load_functions, load_prompts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling"
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )

    return parser.parse_args()


def main() -> int:
    """Initialize and validate the project inputs."""
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Call Me Maybe")
    print(f"Loaded {len(functions)} functions")
    print(f"Loaded {len(prompts)} prompts")

    for function in functions:
        print(f"  Function: {function.name}")

    print(f"Output destination: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
