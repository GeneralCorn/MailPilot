"""Dispatch message for routing between runtime and agents."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class DispatchMessage(BaseModel):
    """Message sent to the runtime for dispatching to an agent."""
    target: str = Field(description="Agent name: planner, researcher, router, evaluator, ranker, worker, etc.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Data for the agent (e.g. email, task, context)")
    correlation_id: str = Field(default="", description="Optional id to correlate request/response")
