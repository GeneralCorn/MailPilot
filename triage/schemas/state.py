from __future__ import annotations
from pydantic import BaseModel, Field

from .enums import Category, Priority
from .message import Message
from .tool import ToolCall, ToolResult


class State(BaseModel):
    """Accumulated pipeline data passed between stages."""
    messages: list[Message] = Field(default_factory=list)
    classifications: dict[str, Category] = Field(default_factory=dict)
    priorities: dict[str, Priority] = Field(default_factory=dict)
    risk_scores: dict[str, float] = Field(default_factory=dict)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    pending_calls: list[ToolCall] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)
    needs_review: list[str] = Field(default_factory=list)
