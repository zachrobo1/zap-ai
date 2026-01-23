# Message Types

Types for multimodal content in messages. These types enable you to pass text and images to your agents.

## Overview

Zap supports multimodal content through these types:

- **`TextContent`**: Plain text content
- **`ImageContent`**: Image content (URL or base64-encoded data)
- **`ContentPart`**: Union type of text or image content
- **`MessageContent`**: Either a string or a list of content parts

## Usage Example

```python
from zap_ai import (
    Zap,
    ZapAgent,
    TextContent,
    ImageContent,
)

# Create agent with vision-capable model
agent = ZapAgent(
    name="VisionAgent",
    prompt="You are a helpful assistant that can see images.",
    model="gpt-4o",
)

zap = Zap(agents=[agent])
await zap.start()

# Send multimodal message with text and image
task = await zap.execute_task(
    agent_name="VisionAgent",
    task=[
        TextContent(type="text", text="What's in this image?"),
        ImageContent(
            type="image_url",
            image_url={"url": "https://example.com/photo.jpg"}
        ),
    ],
)
```

## API Documentation

### TextContent

::: zap_ai.TextContent
    options:
      show_source: false

### ImageContent

::: zap_ai.ImageContent
    options:
      show_source: false

### ContentPart

::: zap_ai.ContentPart
    options:
      show_source: false

### MessageContent

::: zap_ai.MessageContent
    options:
      show_source: false

## See Also

- [Multimodal Vision Guide](../guides/multimodal-vision.md) - Complete guide to using images with agents
- [Core API](core.md) - Main Zap classes
