from typing import Literal, NotRequired, TypedDict

TraceStage = Literal["router", "evaluator", "ranker", "worker", "system"]
TraceLevel = Literal["info", "warning", "error"]


class MailPilotTrace(TypedDict):
    timestamp: str
    stage: TraceStage
    level: TraceLevel
    email_id: str
    event: str
    message: str
    error: NotRequired[str]
