from __future__ import annotations

from typing import Any

import anthropic


def call_tools(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: Any,
    messages: list[dict],
    tools: list[dict],
    force_tool: str | None = None,
    max_tokens: int = 1024,
) -> list[dict] | None:
    """Make an Anthropic messages.create() call requesting structured tool use.

    Returns a list of {"name": str, "input": dict} for each tool_use block in the
    response, or None on API error / refusal. force_tool="X" pins the model to
    that one tool (for single-output agents); leave it None to let the model
    pick freely (multi-tool agents like Worker).
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    if force_tool is not None:
        kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

    try:
        resp = client.messages.create(**kwargs)
    except anthropic.APIError:
        return None

    out: list[dict] = []
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            out.append({"name": block.name, "input": dict(block.input)})
    return out
