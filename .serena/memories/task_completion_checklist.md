# Task Completion Checklist

When completing a coding task, ensure the following steps are performed:

## 1. Code Quality
- [ ] Code follows the project's style conventions (see `code_style_conventions.md`)
- [ ] Type hints are added for all function parameters and return types
- [ ] Docstrings are added for public classes and methods
- [ ] No nested if statements - prefer early returns

## 2. Documentation Updates
**Important**: When making significant changes (new features, API changes, refactoring), update relevant documentation:
- [ ] `README.md` - Update if the change affects user-facing features or usage examples
- [ ] `docs/api/` - Update API reference docs for new/modified public classes and functions
- [ ] `docs/guides/` - Update guide documentation if usage patterns change
- [ ] Docstrings in code should be kept in sync with documentation

## 3. Formatting & Linting
Run these commands and fix any issues:
```bash
# Format code
uv run ruff format .

# Lint and fix
uv run ruff check --fix .
```

## 4. Testing
```bash
# Run unit tests
uv run pytest tests/unit/ -v

# If you added new functionality, ensure there are corresponding tests
```

## 5. Pre-commit Hooks
If pre-commit is installed, hooks run automatically on commit. You can also run manually:
```bash
uv run pre-commit run --all-files
```

## 6. Integration Tests (if applicable)
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
