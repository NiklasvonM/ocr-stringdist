venv:
    rm -rf .venv
    uv venv
    uv sync --all-groups

pytest:
    uv run maturin develop
    uv run pytest --cov=python/ocr_stringdist python/tests

test:
    cargo llvm-cov --features python
    #cargo test --features python

mypy:
    uv run mypy .

lint:
    uv run ruff check . --fix

doc:
    uv run make -C docs html

# Usage: just release v1.0.0
# Make sure to update the version in Cargo.toml first.
release version:
    # Fail if the current branch is not 'main'
    @if [ "$(git symbolic-ref --short HEAD)" != "main" ]; then \
        echo "Error: Must be on 'main' branch to release."; \
        exit 1; \
    fi

    # Fail if the working directory is not clean
    @if ! git diff --quiet --exit-code; then \
        echo "Error: Working directory is not clean. Commit or stash changes before releasing."; \
        exit 1; \
    fi

    git tag -a {{version}} -m "Release version {{version}}"
    git push origin {{version}}

    @echo "Successfully tagged and pushed version {{version}}"
