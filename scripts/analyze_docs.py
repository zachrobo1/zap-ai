#!/usr/bin/env python3
"""
Pre-push documentation analysis.

Analyzes branch changes to determine if .serena memories or docs need updating.
"""

import json
import os
import subprocess
import sys

import litellm
from pydantic import BaseModel


class AnalysisResult(BaseModel):
    needs_update: bool
    summary: str
    memory_suggestions: list[dict]  # {file, reason, changes}
    doc_suggestions: list[dict]  # {file, reason, changes}


MEMORY_DESCRIPTIONS = {
    "project_overview.md": "Project architecture, modules, classes, package structure",
    "suggested_commands.md": "Development workflow commands",
    "code_style_conventions.md": "Code quality guidelines",
    "task_completion_checklist.md": "Completion requirements",
}

DOC_STRUCTURE = {
    "docs/getting-started/": "Installation and quickstart",
    "docs/guides/": "Feature guides (dynamic-prompts, multi-agent, approval-workflows)",
    "docs/api/": "API reference (core, conversation, tracing, worker, exceptions)",
}


def get_branch_diff() -> tuple[str, list[str]]:
    """Get diff and changed files for current branch vs main."""
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()

    if branch == "main":
        print("On main branch, skipping analysis.")
        sys.exit(0)

    merge_base = subprocess.check_output(["git", "merge-base", "main", "HEAD"], text=True).strip()

    diff = subprocess.check_output(["git", "diff", merge_base, "HEAD"], text=True)

    files = (
        subprocess.check_output(["git", "diff", "--name-only", merge_base, "HEAD"], text=True)
        .strip()
        .split("\n")
    )

    return diff, [f for f in files if f]


def should_skip(changed_files: list[str]) -> bool:
    """Skip for docs-only or CI-only changes."""
    if all(f.startswith("docs/") or f.endswith(".md") for f in changed_files):
        return True
    if all(f.startswith(".github/") for f in changed_files):
        return True
    return False


def analyze(diff: str, changed_files: list[str]) -> AnalysisResult:
    """Use Claude to analyze changes."""
    already_updated_docs = [f for f in changed_files if f.startswith("docs/")]
    already_updated_memories = [f for f in changed_files if f.startswith(".serena/memories/")]

    prompt = f"""Analyze these code changes and determine if documentation needs updating.

## .serena Memory Files
{MEMORY_DESCRIPTIONS}
Already updated in this branch: {already_updated_memories or "None"}

## Documentation Structure
{DOC_STRUCTURE}
Already updated in this branch: {already_updated_docs or "None"}

## Changed Files
{chr(10).join(f"- {f}" for f in changed_files)}

## Diff
```diff
{diff[:15000]}
```

Respond with JSON:
{{
  "needs_update": boolean,
  "summary": "Brief summary",
  "memory_suggestions": [{{"file": "...", "reason": "...", "changes": "..."}}],
  "doc_suggestions": [{{"file": "...", "reason": "...", "changes": "..."}}]
}}

Be conservative - only suggest updates for significant changes. Skip files already updated."""

    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return AnalysisResult(**json.loads(text))


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping documentation analysis")
        sys.exit(0)

    diff, changed_files = get_branch_diff()

    if not changed_files:
        print("No changes to analyze.")
        sys.exit(0)

    if should_skip(changed_files):
        print("Documentation/CI only changes, skipping analysis.")
        sys.exit(0)

    print("Analyzing documentation needs...")
    result = analyze(diff, changed_files)

    if not result.needs_update:
        print("No documentation updates needed.")
        sys.exit(0)

    print("\nDocumentation Review Suggested\n")
    print(f"Summary: {result.summary}\n")

    if result.memory_suggestions:
        print("## .serena Memory Updates")
        for s in result.memory_suggestions:
            print(f"  - {s['file']}: {s['reason']}")
            print(f"    -> {s['changes']}\n")

    if result.doc_suggestions:
        print("## Documentation Updates")
        for s in result.doc_suggestions:
            print(f"  - {s['file']}: {s['reason']}")
            print(f"    -> {s['changes']}\n")

    # Exit 0 (non-blocking) - just a warning
    sys.exit(0)


if __name__ == "__main__":
    main()
