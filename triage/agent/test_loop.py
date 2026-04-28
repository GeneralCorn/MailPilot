from __future__ import annotations

from unittest.mock import patch

from triage.agent import loop as loop_mod
from triage.agent.evaluator import EvaluatorResult
from triage.schemas import Category, Message, State


def _msg(eid: str = "e1") -> Message:
    return Message(id=eid, subject="x", sender="a@b", body_plain="hi")


def test_loop_converges_on_first_iteration():
    state = State(messages=[_msg()])

    def fake_route(email, state, *, feedback=None):
        state.classifications[email.id] = Category.WORK
        state.confidence_scores[email.id] = 0.9

    def fake_eval(email, state):
        return EvaluatorResult(
            final_category=Category.WORK,
            confidence=0.9,
            risk_score=0.1,
            needs_review=False,
            explanation="ok",
        )

    with patch.object(loop_mod, "route", side_effect=fake_route), patch.object(
        loop_mod, "evaluate", side_effect=fake_eval
    ):
        result = loop_mod.route_eval_loop(_msg(), state)

    assert result.final_category == Category.WORK
    assert state.iteration_counts["e1"] == 1
    assert "e1" not in state.needs_review


def test_loop_converges_on_second_iteration_via_feedback():
    state = State(messages=[_msg()])
    seen_feedbacks: list[str | None] = []

    def fake_route(email, state, *, feedback=None):
        seen_feedbacks.append(feedback)
        state.classifications[email.id] = Category.PERSONAL
        state.confidence_scores[email.id] = 0.95 if feedback else 0.4

    eval_calls = {"n": 0}

    def fake_eval(email, state):
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return EvaluatorResult(
                final_category=Category.PERSONAL,
                confidence=0.4,
                risk_score=0.05,
                needs_review=True,
                explanation="confidence too low, look again",
            )
        return EvaluatorResult(
            final_category=Category.PERSONAL,
            confidence=0.95,
            risk_score=0.05,
            needs_review=False,
            explanation="ok now",
        )

    with patch.object(loop_mod, "route", side_effect=fake_route), patch.object(
        loop_mod, "evaluate", side_effect=fake_eval
    ):
        result = loop_mod.route_eval_loop(_msg(), state)

    assert result.confidence == 0.95
    assert state.iteration_counts["e1"] == 2
    assert seen_feedbacks == [None, "confidence too low, look again"]
    assert "e1" not in state.needs_review


def test_loop_hits_max_iterations_then_flags():
    state = State(messages=[_msg()])

    def fake_route(email, state, *, feedback=None):
        state.classifications[email.id] = Category.UNCLASSIFIED
        state.confidence_scores[email.id] = 0.3

    def fake_eval(email, state):
        return EvaluatorResult(
            final_category=Category.UNCLASSIFIED,
            confidence=0.3,
            risk_score=0.1,
            needs_review=True,
            explanation="still unclear",
        )

    with patch.object(loop_mod, "route", side_effect=fake_route), patch.object(
        loop_mod, "evaluate", side_effect=fake_eval
    ):
        loop_mod.route_eval_loop(_msg(), state)

    assert state.iteration_counts["e1"] == loop_mod.MAX_ROUTER_EVAL_ITERATIONS
    assert "e1" in state.needs_review


def test_loop_breaks_on_legit_risk_without_re_routing():
    state = State(messages=[_msg()])
    route_calls = {"n": 0}

    def fake_route(email, state, *, feedback=None):
        route_calls["n"] += 1
        state.classifications[email.id] = Category.RISK
        state.confidence_scores[email.id] = 0.92

    def fake_eval(email, state):
        return EvaluatorResult(
            final_category=Category.RISK,
            confidence=0.92,
            risk_score=0.95,
            needs_review=True,
            explanation="phishing",
        )

    with patch.object(loop_mod, "route", side_effect=fake_route), patch.object(
        loop_mod, "evaluate", side_effect=fake_eval
    ):
        result = loop_mod.route_eval_loop(_msg(), state)

    assert route_calls["n"] == 1
    assert result.final_category == Category.RISK
    assert "e1" in state.needs_review
