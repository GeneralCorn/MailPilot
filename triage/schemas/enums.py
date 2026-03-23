from enum import Enum


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
