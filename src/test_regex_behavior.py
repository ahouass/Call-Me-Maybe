
"""Verify that generated regex performs the requested replacement."""

import json
import re
from pathlib import Path


def main() -> None:
    """Test the vowel replacement from the benchmark."""
    path = Path("data/output/benchmark_regex_results.json")
    results = json.loads(path.read_text(encoding="utf-8"))

    parameters = results[0]["parameters"]

    actual = re.sub(
        parameters["regex"],
        parameters["replacement"],
        parameters["source_string"]
    )

    expected = "Pr*gr*mm*ng *s f*n"

    print("Regex:", repr(parameters["regex"]))
    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected, "Incorrect regex behavior"


if __name__ == "__main__":
    main()
