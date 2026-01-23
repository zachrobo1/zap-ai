"""Temporal workflow sandbox configuration.

Temporal's workflow sandbox restricts module imports to ensure determinism.
Some modules require passthrough configuration to work correctly.
"""

from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

# Production environment restrictions
# - beartype: Runtime type checking causes circular import issues
# - httpx: Uses urllib.request which is restricted by default
# - litellm: Uses thread-local objects incompatible with sandbox
# - urllib: Standard library module needed by httpx
production_restrictions = SandboxRestrictions.default.with_passthrough_modules(
    "beartype",
    "httpx",
    "litellm",
    "urllib",
)


def create_production_runner() -> SandboxedWorkflowRunner:
    """Create sandbox runner for production environments.

    Returns:
        SandboxedWorkflowRunner with production module passthroughs.
    """
    return SandboxedWorkflowRunner(restrictions=production_restrictions)


__all__ = [
    "production_restrictions",
    "create_production_runner",
]
