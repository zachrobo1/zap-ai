# Multimodal Vision Support

Zap supports multimodal content, allowing you to send images and other visual content to vision-capable LLM models. This guide covers how to use these features.

## Prerequisites

- A vision-capable model (e.g., `anthropic/claude-sonnet-4-20250514`, `gpt-4o`)
- Images accessible via URL or base64-encoded

## Basic Usage

### Sending Images with Tasks

Use `TextContent` and `ImageContent` to create multimodal tasks:

```python
from zap_ai import Zap, ZapAgent, TextContent, ImageContent

# Create a vision-capable agent
vision_agent = ZapAgent(
    name="VisionAgent",
    prompt="You are a helpful assistant that can analyze images.",
    model="anthropic/claude-sonnet-4-20250514",  # Vision-capable model
)

zap = Zap(agents=[vision_agent])
await zap.start()

# Send an image with your task
task = await zap.execute_task(
    agent_name="VisionAgent",
    task_content=[
        TextContent(text="What do you see in this image?"),
        ImageContent.from_url("https://example.com/photo.jpg"),
    ],
)
```

### Image Sources

Images can be provided from two sources:

**URL (recommended for remote images):**
```python
img = ImageContent.from_url("https://example.com/image.png")
```

**Base64 (for local images):**
```python
import base64

# Read and encode a local image
with open("local_image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

img = ImageContent.from_base64(image_data, mime_type="image/png")
```

### Multiple Images

You can include multiple images in a single task:

```python
task = await zap.execute_task(
    agent_name="VisionAgent",
    task_content=[
        TextContent(text="Compare these two images:"),
        ImageContent.from_url("https://example.com/image1.jpg"),
        ImageContent.from_url("https://example.com/image2.jpg"),
    ],
)
```

## Model Capability Validation

Zap automatically validates that your agent's model supports vision before processing image content. If you try to send images to a non-vision model, you'll get a `VisionNotSupportedError`:

```python
from zap_ai import VisionNotSupportedError

# This agent uses a non-vision model
text_agent = ZapAgent(
    name="TextAgent",
    prompt="You are a text-only assistant.",
    model="gpt-3.5-turbo",  # Does NOT support vision
)

try:
    task = await zap.execute_task(
        agent_name="TextAgent",
        task_content=[
            TextContent(text="Describe this:"),
            ImageContent.from_url("https://example.com/photo.jpg"),
        ],
    )
except VisionNotSupportedError as e:
    print(f"Model doesn't support vision: {e}")
```

## Advanced Options

### Image Detail Level

For supported models, you can specify the detail level for image processing:

```python
# High detail for detailed analysis
img = ImageContent.from_url(
    "https://example.com/document.png",
    detail="high"  # Options: "auto", "low", "high"
)

# Low detail for faster processing
img = ImageContent.from_url(
    "https://example.com/thumbnail.png",
    detail="low"
)
```

### Explicit MIME Type

For cloud storage URLs (like `gs://`), you may need to specify the format explicitly:

```python
img = ImageContent(
    url="gs://my-bucket/image.webp",
    format="image/webp"
)
```

## Working with Messages

The `Message` class provides helper methods for multimodal content:

```python
from zap_ai.llm.message_types import Message

# Create a user message with images
msg = Message.user_with_images(
    "What's in this image?",
    [ImageContent.from_url("https://example.com/photo.jpg")]
)

# Check if a message has images
if msg.has_images():
    print("Message contains images")

# Extract text content (ignoring images)
text = msg.get_text_content()
```

## Sub-Agent Communication

When using sub-agents, you can pass multimodal content through the `message_agent` tool:

```python
# The parent agent can pass images to sub-agents
# via the message_agent tool with multimodal content

coordinator = ZapAgent(
    name="Coordinator",
    prompt="You coordinate with specialized vision agents.",
    model="anthropic/claude-sonnet-4-20250514",
    sub_agents=["ImageAnalyzer"],
)
```

The sub-agent will receive the full multimodal content, including any images.

## Best Practices

1. **Use vision-capable models**: Check LiteLLM's documentation for supported vision models
2. **Prefer URLs for remote images**: Base64 encoding increases payload size significantly
3. **Consider detail levels**: Use `"low"` for thumbnails, `"high"` for documents requiring text extraction
4. **Handle errors gracefully**: Catch `VisionNotSupportedError` when model support is uncertain

## Example

See the complete example in `examples/multimodal_vision/main.py`:

```bash
cd examples/multimodal_vision
python main.py
```
