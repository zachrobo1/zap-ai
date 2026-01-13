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
- [ ] `docs/guides/` - Update guide documentation if usage patterns change (approval-workflows.md, multi-agent.md, etc.)
- [ ] `docs/index.md` - Update feature highlights if adding major features
- [ ] `mkdocs.yml` - Add new guide pages to navigation
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

## 5. Documentation Build
Verify docs build successfully:
```bash
uv run mkdocs build --strict
```

## 6. Pre-commit Hooks
If pre-commit is installed, hooks run automatically on commit. You can also run manually:
```bash
uv run pre-commit run --all-files
```

## 7. Integration Tests (if applicable)
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
uv run mkdocs build --strict
```

All checks must pass before the task is considered complete.

## PR Guidelines
When creating PRs:
- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- `feat:` triggers minor version bump via release-please
- Include a Summary section with bullet points
- Include a Test plan section with checkboxes
- End with "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
