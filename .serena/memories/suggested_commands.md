# Development Commands

This project uses **uv** as the package manager. All commands should be run from the project root.

## Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

## Testing
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run a specific test file
uv run pytest tests/unit/core/test_agent.py -v

# Run a specific test
uv run pytest tests/unit/core/test_agent.py::TestZapAgentCreation::test_minimal_agent -v

# Run with coverage
uv run pytest tests/unit/ --cov=src/zap_ai --cov-report=term-missing

# Run integration tests (requires Temporal server running)
temporal server start-dev  # In separate terminal
uv run pytest tests/integration/ -v
```

## Linting & Formatting
```bash
# Check formatting (no changes)
uv run ruff format --check .

# Format code
uv run ruff format .

# Check linting (no auto-fix)
uv run ruff check .

# Lint and auto-fix
uv run ruff check --fix .

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Building
```bash
# Build the package
uv build

# Check package contents
tar -tzf dist/zap_ai-*.tar.gz
```

## Temporal Server (for local development)
```bash
# Start local Temporal server (requires Temporal CLI)
temporal server start-dev

# Or with headless mode (no UI)
temporal server start-dev --headless
```

## Git Commands (Darwin/macOS)
```bash
# Standard git commands work as expected
git status
git add .
git commit -m "message"
git push
git pull
git log --oneline -10
git diff
```

## System Commands (Darwin/macOS)
```bash
# List directory contents
ls -la

# Find files
find . -name "*.py" -type f

# Search in files
grep -r "pattern" src/

# View file
cat filename
```
