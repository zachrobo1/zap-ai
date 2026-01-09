# Code Style and Conventions

## General Guidelines
- **Python 3.11+** is required
- Line length: **100 characters** (configured in pyproject.toml)
- Use **ruff** for linting and formatting

## Linting Rules (ruff)
The following rule sets are enabled:
- `E` - pycodestyle errors
- `F` - pyflakes
- `I` - isort (import sorting)
- `W` - pycodestyle warnings

## Type Hints
- **Always use type hints** for function parameters and return types
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- For complex types, import from `typing` as needed

```python
def process_items(items: list[str], count: int | None = None) -> dict[str, Any]:
    ...
```

## Docstrings
- Use **triple double quotes** for docstrings
- Include docstrings for:
  - All public classes (with Attributes section)
  - All public methods/functions
  - Complex private functions
- Use descriptive attribute documentation in Pydantic models via `Field(description=...)`

Example class docstring:
```python
class MyClass:
    """
    Brief description of the class.

    More detailed explanation if needed.

    Example:
        ```python
        obj = MyClass(...)
        ```

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.
    """
```

## Naming Conventions
- **Classes**: PascalCase (e.g., `ZapAgent`, `TaskStatus`)
- **Functions/Methods**: snake_case (e.g., `execute_task`, `get_status`)
- **Variables**: snake_case (e.g., `agent_name`, `max_iterations`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`)
- **Private**: Single underscore prefix (e.g., `_internal_method`)

## Control Flow
- **Prefer early returns** over nested if statements
- Check for negative conditions first, then return/continue

```python
# Good
def process(item):
    if not item:
        return None
    if not item.is_valid:
        return None
    return item.process()

# Avoid
def process(item):
    if item:
        if item.is_valid:
            return item.process()
    return None
```

## Imports
- Standard library imports first
- Third-party imports second
- Local imports third
- Separate groups with blank lines
- ruff/isort handles sorting automatically

## Testing
- Test files: `test_<module>.py`
- Test classes: `TestClassName`
- Test methods: `test_<behavior>` (e.g., `test_empty_name_rejected`)
- Use descriptive docstrings for test methods
- Use pytest fixtures for shared setup
- Mark async tests appropriately (pytest-asyncio with auto mode)

## Pydantic Models
- Use `Field()` for validation and documentation
- Use `field_validator` for custom validation
- Include `model_config` when needed (e.g., `arbitrary_types_allowed`)

## Exception Handling Pattern
- All custom exceptions are consolidated in `src/zap_ai/exceptions.py`
- All exceptions inherit from `ZapError` base class
- Import exceptions from `zap_ai.exceptions` (not from individual modules)
- Exception classes are organized by domain:
  - Core: `ZapConfigurationError`, `ZapNotStartedError`, `AgentNotFoundError`, `TaskNotFoundError`
  - MCP/Tool: `ToolNotFoundError`, `ToolExecutionError`, `ClientConnectionError`, `SchemaConversionError`
  - LLM: `LLMProviderError`

```python
# Good - import from exceptions module
from zap_ai.exceptions import ToolNotFoundError, ToolExecutionError

# Avoid - defining exceptions in multiple places
class ToolNotFoundError(Exception):  # Don't do this
    pass
```

## Separation of Concerns
- **Validation logic** extracted to `core/validation.py` (not in main classes)
- **Conversation parsing** in `conversation/` module (not in Task class)
- **Shared utilities** (like `parse_tool_arguments`) in `utils.py`
- Business logic classes delegate to extracted functions for complex operations
