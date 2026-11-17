
.PHONY: all install sync run debug clean lint check

all: run

install:
	uv sync

sync: install

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

check: lint

clean:
	rm -rf .mypy_cache .pytest_cache
	find src -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src -type f -name "*.pyc" -delete
