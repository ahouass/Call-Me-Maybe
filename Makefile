
.PHONY: all run lint typecheck check sync clean

all: run

sync:
	uv sync

run:
	uv run python -m src

lint:
	uv run flake8 src

typecheck:
	uv run python -m mypy src

check: lint typecheck
	@echo "All code-quality checks passed!"

clean:
	find src -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src -type f -name "*.pyc" -delete
