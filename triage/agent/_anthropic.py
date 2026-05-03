from __future__ import annotations

from typing import Any

import anthropic

from . import _usage


def call_tools(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: Any,
    messages: list[dict],
    tools: list[dict],
    force_tool: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
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
        "temperature": temperature,
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

    usage = getattr(resp, "usage", None)
    if usage is not None:
        _usage.record(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )

    out: list[dict] = []
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            out.append({"name": block.name, "input": dict(block.input)})
    return out


def call_json(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: Any,
    user: str,
    output_schema: dict,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict | None:
    """One-shot structured output via a single forced tool whose schema is the output schema."""
    tool = {"name": "submit", "description": "Submit the structured output.", "input_schema": output_schema}
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "submit"},
    }
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.APIError:
        return None

    usage = getattr(resp, "usage", None)
    if usage is not None:
        _usage.record(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )

    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None
