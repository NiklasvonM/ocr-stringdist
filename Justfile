venv:
    rm -rf .venv
    uv venv
    uv sync

pytest:
    uv run maturin develop
    uv run pytest

test:
    cargo test

mypy:
    uv run mypy .

lint:
    uv run ruff check . --fix

doc:
    uv run make -C docs html
