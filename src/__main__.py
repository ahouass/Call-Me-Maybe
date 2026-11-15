
"""Main entry point for Call Me Maybe."""

import argparse
import json
import sys
import time
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .function_decoder import FunctionDecoder
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
    """Generate function calls and save them as JSON."""
    args = parse_args()
    start = time.perf_counter()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

        if not functions:
            raise ValueError("No function definitions provided")

        print(f"Loaded {len(functions)} functions")
        print(f"Loaded {len(prompts)} prompts")

        model = Small_LLM_Model()
        decoder = FunctionDecoder(model)

        results = []

        for index, item in enumerate(prompts, start=1):
            print(
                f"Processing {index}/{len(prompts)}: "
                f"{item.prompt}",
                flush=True
            )

            prompt_start = time.perf_counter()

            result = decoder.generate(
                item.prompt,
                functions
            )

            elapsed_prompt = time.perf_counter() - prompt_start

            print(
                f"Prompt {index} took {elapsed_prompt:.2f} seconds",
                flush=True
            )

            results.append({
                "prompt": item.prompt,
                "name": result["name"],
                "parameters": result["parameters"]
            })

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False
            ) + "\n",
            encoding="utf-8"
        )

    except (ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start

    print(f"Saved {len(results)} results to {output_path}")
    print(f"Total time: {elapsed:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
