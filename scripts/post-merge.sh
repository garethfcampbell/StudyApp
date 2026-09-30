#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the locked Python environment ready and run the project's regression tests.
uv sync --frozen --no-install-project
uv run --frozen --no-sync python -m unittest discover -s tests -v