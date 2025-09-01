venv:
    rm -rf .venv
    uv venv
    uv sync --all-groups

pytest:
    uv run maturin develop
    uv run pytest --cov=python/ocr_stringdist tests

test:
    cargo llvm-cov
    #cargo test

mypy:
    uv run mypy .

lint:
    uv run ruff check . --fix

doc:
    uv run make -C docs html
