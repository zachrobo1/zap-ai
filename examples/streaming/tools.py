"""Local MCP tools for the streaming example.

These tools complement the mcp-server-fetch for text analysis tasks.
"""

import ast
import operator
import re

from fastmcp import FastMCP

mcp = FastMCP("Analysis Tools")


@mcp.tool()
def analyze_text(text: str) -> str:
    """Analyze text for word count, reading level, and key statistics."""
    # Word count
    words = text.split()
    word_count = len(words)

    # Sentence count (simple heuristic)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Average word length
    total_chars = sum(len(word) for word in words)
    avg_word_length = total_chars / word_count if word_count > 0 else 0

    # Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Paragraph count
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    # Character count (excluding spaces)
    char_count = len(text.replace(" ", "").replace("\n", ""))

    # Flesch Reading Ease approximation (simplified)
    syllables_per_word = avg_word_length / 3  # rough approximation
    reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * syllables_per_word)
    reading_ease = max(0, min(100, reading_ease))

    if reading_ease >= 80:
        reading_level = "Easy (Grade 6)"
    elif reading_ease >= 60:
        reading_level = "Standard (Grade 8-9)"
    elif reading_ease >= 40:
        reading_level = "Fairly Difficult (Grade 10-12)"
    else:
        reading_level = "Difficult (College level)"

    return f"""Text Analysis Results:
- Words: {word_count:,}
- Sentences: {sentence_count:,}
- Paragraphs: {paragraph_count:,}
- Characters: {char_count:,}
- Average word length: {avg_word_length:.1f} characters
- Average sentence length: {avg_sentence_length:.1f} words
- Reading level: {reading_level} (Flesch score: {reading_ease:.0f})"""


@mcp.tool()
def summarize_bullet_points(text: str, max_points: int = 5) -> str:
    """Extract key bullet points from text."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

    # Score sentences by importance (simple heuristic)
    scored = []
    for sentence in sentences:
        score = 0
        # Longer sentences often have more content
        score += min(len(sentence) / 100, 2)
        # Sentences with numbers often contain facts
        if re.search(r"\d+", sentence):
            score += 1
        # Sentences with key indicator words
        indicators = ["is", "are", "was", "were", "first", "main", "important", "key"]
        if any(word in sentence.lower() for word in indicators):
            score += 0.5
        # Avoid sentences that are too short
        if len(sentence) < 50:
            score -= 1
        scored.append((score, sentence))

    # Sort by score and take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:max_points]]

    if not top_sentences:
        return "No key points could be extracted from the text."

    bullets = "\n".join(f"- {s}" for s in top_sentences)
    return f"Key Points:\n{bullets}"


# Safe operators for calculate
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Safely evaluate an AST node containing only arithmetic operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@mcp.tool()
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Supports: +, -, *, /, //, %, **
    Examples: "2 + 2", "10 * 5", "2 ** 10", "100 / 3"
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode="eval")
        # Safely evaluate
        result = _safe_eval(tree)
        # Format result nicely
        if result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {result:.6g}"
    except (SyntaxError, ValueError) as e:
        return f"Error: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"


if __name__ == "__main__":
    mcp.run()
