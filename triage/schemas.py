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
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Status(str, Enum):
    PENDING = "pending"
    DONE = "done"
    PARTIAL_DONE = "partial_done"
    FLAGGED = "flagged"
    AWAITING_APPROVAL = "awaiting_approval"


class Action(str, Enum):
    LABEL = "label"
    FLAG = "flag"
    ARCHIVE = "archive"
    REPLY_DRAFT = "reply_draft"
    CALENDAR = "calendar"
    ESCALATE = "escalate"
    SUMMARIZE = "summarize"
    SEND_EMAIL = "send_email"
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
    category: str = "unclassified"

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
    needs_review: list[str] = Field(default_factory=list)
    priority_queue: list[tuple[str, float]] = Field(default_factory=list)
    email_status: dict[str, Status] = Field(default_factory=dict)
    worker_actions: dict[str, list[ToolCall]] = Field(default_factory=dict)
    sub_action_results: dict[str, list[ToolResult]] = Field(default_factory=dict)
    iteration_counts: dict[str, int] = Field(default_factory=dict)
    # router's first-iteration output, preserved before evaluator can override classifications
    router_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # actions Worker proposed but defers to user approval (calendar / send_email)
    proposed_actions: dict[str, list[ToolCall]] = Field(default_factory=dict)


