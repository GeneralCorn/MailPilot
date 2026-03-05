"""Pydantic schemas for the MailPilot agent pipeline."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ──

class Category(str, Enum):
    MARKETING = "marketing"
    PERSONAL = "personal"
    WORK = "work"
    RISK = "risk"
    BILLING = "billing"
    UNCLASSIFIED = "unclassified"


class Priority(str, Enum):
    URGENT = "urgent"
    IMPORTANT = "important"
    NORMAL = "normal"
    LOW = "low"
    MINIMAL = "minimal"


class Status(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class Action(str, Enum):
    LABEL = "label"
    FLAG = "flag"
    ARCHIVE = "archive"
    REPLY_DRAFT = "reply_draft"
    CALENDAR = "calendar"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


# ── Pipeline Schemas ──

class Message(BaseModel):
    """Email metadata passed into LLM context windows."""
    id: str = ""
    subject: str
    sender: str
    sender_name: str = ""
    recipient: str = ""
    body_html: str = ""
    body_plain: str = ""
    snippet: str = ""
    received_at: datetime | None = None
    thread_id: str = ""
    source: str = "manual"

class AgentMessage(BaseModel):
    """Minimal prompt message passed by the orchestrator."""
    role: str
    content: str

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
