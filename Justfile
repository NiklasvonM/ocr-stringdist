venv:
    rm -rf .venv
    uv venv
    uv sync

pytest:
    maturin develop
    uv run pytest

test:
    cargo test

mypy:
    uv run mypy .
