from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

from .enums import Action


class Tool(BaseModel):
    """Definition of an available action + its parameters."""
    name: Action
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """LLM's request to execute an action."""
    tool: Action
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolResult(BaseModel):
    """Outcome of executing a tool call."""
    tool: Action
    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
