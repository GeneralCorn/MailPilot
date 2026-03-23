from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


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
