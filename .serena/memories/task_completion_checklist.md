# Task Completion Checklist

When completing a coding task, ensure the following steps are performed:

## 1. Code Quality
- [ ] Code follows the project's style conventions (see `code_style_conventions.md`)
- [ ] Type hints are added for all function parameters and return types
- [ ] Docstrings are added for public classes and methods
- [ ] No nested if statements - prefer early returns

## 2. Formatting & Linting
Run these commands and fix any issues:
```bash
# Format code
uv run ruff format .

# Lint and fix
uv run ruff check --fix .
```

## 3. Testing
```bash
# Run unit tests
uv run pytest tests/unit/ -v

# If you added new functionality, ensure there are corresponding tests
```

## 4. Pre-commit Hooks
If pre-commit is installed, hooks run automatically on commit. You can also run manually:
```bash
uv run pre-commit run --all-files
```

## 5. Integration Tests (if applicable)
If your changes affect Temporal workflows or integrations:
```bash
# Start Temporal server first
temporal server start-dev

# Run integration tests
uv run pytest tests/integration/ -v
```

## Summary Commands
Quick checklist to run before considering a task complete:
```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest tests/unit/ -v
```

All checks must pass before the task is considered complete.
