from __future__ import annotations

from triage.agent.evaluator import EvaluatorResult
from triage.schemas import Category, Message, State

from . import mailpilot_full


def passthrough_evaluator(email: Message, state: State) -> EvaluatorResult:
    """No-op evaluator: always reports the router's choice as fully reliable."""
    cat = state.classifications.get(email.id, Category.UNCLASSIFIED)
    return EvaluatorResult(
        final_category=cat,
        confidence=1.0,
        risk_score=0.0,
        needs_review=False,
        explanation="evaluator bypassed",
    )


def run(emails: list[Message]) -> list[dict]:
    return mailpilot_full.run(emails, evaluator_fn=passthrough_evaluator)
