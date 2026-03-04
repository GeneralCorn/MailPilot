from typing import Literal, TypedDict

from .EmailRecord import EmailRecord
from .MailPilotTrace import MailPilotTrace


EmailCategory = Literal[
    "marketing",
    "personal",
    "work",
    "risk",
    "billing",
    "unclassified",
]


class MailPilotRuntime(TypedDict):
    routed_category: EmailCategory
    confidence: float
    safety_flags: list[str]
    priority_score: float
    needs_review: bool


class MailPilotState(TypedDict):
    emails: list[EmailRecord]
    need_human_review: list[str]
    human_approved: list[str]
    runtime: dict[str, MailPilotRuntime]
    trace: list[MailPilotTrace]
