from typing import Literal, NotRequired, TypedDict

EmailCategory = Literal[
    "marketing",
    "personal",
    "work",
    "risk",
    "billing",
    "unclassified",
]

EmailStatus = Literal[
    "pending",
    "processing",
    "done",
    "failed",
    "needs_review",
]


class EmailRecord(TypedDict):
    id: str
    subject: str
    sender: str
    received_at: str
    body: str
    category: EmailCategory
    status: EmailStatus
    confidence: float
    priority: int
    risk_score: float
    explanation: NotRequired[str]
    artifacts: NotRequired[dict[str, object]]
