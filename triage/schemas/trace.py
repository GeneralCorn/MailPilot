from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from .tool import ToolCall, ToolResult


class Trace(BaseModel):
    """Full audit log of a single pipeline run."""
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    input_count: int = 0
    stages: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
