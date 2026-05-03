from __future__ import annotations

from typing import Callable

from triage.schemas import Category, Message, State

from .evaluator import (
    EvaluatorResult,
    HIGH_RISK_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    evaluate,
)
from .router import route

MAX_ROUTER_EVAL_ITERATIONS = 3

EvaluatorFn = Callable[[Message, State], EvaluatorResult]


def _is_reliable(result: EvaluatorResult) -> bool:
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
        return False
    if result.risk_score > HIGH_RISK_THRESHOLD:
        # legitimately risky — don't re-route, but also not "reliable" on the safe path
        return result.final_category == Category.RISK
    return True


def route_eval_loop(
    email: Message, state: State, *, evaluator_fn: EvaluatorFn | None = None
) -> EvaluatorResult:
    feedback: str | None = None
    iterations = 0
    result: EvaluatorResult | None = None

    while iterations < MAX_ROUTER_EVAL_ITERATIONS:
        iterations += 1
        route(email, state, feedback=feedback)
        if email.id not in state.router_outputs:
            state.router_outputs[email.id] = {
                "category": state.classifications.get(email.id, Category.UNCLASSIFIED).value,
                "confidence": state.confidence_scores.get(email.id, 0.0),
            }
        # resolve at call time so test patches on `evaluate` still take effect
        result = (evaluator_fn or evaluate)(email, state)

        if _is_reliable(result):
            break
        feedback = result.explanation

    state.iteration_counts[email.id] = iterations

    assert result is not None
    if (
        iterations >= MAX_ROUTER_EVAL_ITERATIONS and not _is_reliable(result)
    ) or result.needs_review:
        if email.id not in state.needs_review:
            state.needs_review.append(email.id)

    return result
