#!/bin/zsh

# Provision the dev environment with uv: .venv + pyproject.toml (uv sync).
# (.venv, not the old .env: the pre-commit pytest hook and the existing
# uv-managed .venv both expect .venv.)

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ -d ".venv" ]; then
    echo "Environment directory '.venv' already exists."
else
    echo "Creating Python virtual environment in '.venv'..."
    # 3.10: matches CI (check.yml) and the py3.10 SageMaker containers.
    if ! uv venv .venv --python 3.10; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

echo "Installing dependencies (uv sync from pyproject.toml)..."
uv sync --python .venv/bin/python
echo "Dependencies installed successfully."

echo "Activate with: source .venv/bin/activate"
