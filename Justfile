venv:
    rm -rf .venv
    python3 -m venv .venv
    . .venv/bin/activate
    .venv/bin/pip install wheel pytest maturin icecream ruff

pytest:
    maturin develop
    .venv/bin/pytest

test:
    cargo test
