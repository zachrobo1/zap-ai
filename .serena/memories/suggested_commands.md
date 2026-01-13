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

## Documentation
```bash
# Build docs (with strict mode to catch errors)
uv run mkdocs build --strict

# Serve docs locally for preview
uv run mkdocs serve

# Docs are auto-deployed to GitHub Pages on push to main
# Live at: https://zachrobo1.github.io/zap-ai/
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

# Temporal UI available at http://localhost:8233
```

## GitHub CLI (gh)
```bash
# Create a PR
gh pr create --title "feat: description" --body "..."

# View PR
gh pr view <number>

# Check PR status
gh pr status

# List open PRs
gh pr list
```

## Git Workflow
```bash
# Standard git commands
git status
git add .
git commit -m "feat: description"  # Use conventional commits
git push

# View recent commits
git log --oneline -10

# View diff vs main
git diff main

# Branch from main
git checkout main && git pull && git checkout -b feature/my-feature
```

## Conventional Commits
Use these prefixes for commits (triggers release-please versioning):
- `feat:` - New feature (minor version bump)
- `fix:` - Bug fix (patch version bump)
- `docs:` - Documentation only
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks
