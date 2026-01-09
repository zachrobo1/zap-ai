# Installation

## Install Zap

Install from PyPI:

```bash
pip install zap-ai
```

Or with uv:

```bash
uv add zap-ai
```

### Optional Dependencies

For Langfuse tracing support:

```bash
pip install zap-ai[langfuse]
```

## Prerequisites

### 1. Temporal Server

Zap requires a running Temporal cluster. For local development:

```bash
# Using the Temporal CLI (requires Docker)
temporal server start-dev
```

This starts a local Temporal server at `localhost:7233`.

For production, consider [Temporal Cloud](https://temporal.io/cloud) or self-hosting.

### 2. LLM API Keys

Set environment variables for your LLM provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
```

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for the full list of supported providers.

## Verify Installation

```python
import zap_ai

print(zap_ai.__version__)
```

## Next Steps

Continue to [Quick Start](quickstart.md) to build your first agent.
