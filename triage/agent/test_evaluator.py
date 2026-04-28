from unittest.mock import patch

from triage.agent import evaluator as evaluator_mod
from triage.agent._test_helpers import fake_anthropic_with_tool_calls
from triage.schemas import Category, Message, State


def _msg() -> Message:
    return Message(id="e1", subject="x", sender="a@b", body_plain="...")


def _state_with_router(category: Category, confidence: float) -> State:
    s = State(messages=[_msg()])
    s.classifications["e1"] = category
    s.confidence_scores["e1"] = confidence
    return s


def test_evaluator_writes_state_and_returns_result():
    state = _state_with_router(Category.WORK, 0.9)
    factory = fake_anthropic_with_tool_calls([
        [("verify_classification", {
            "reasoning": "looks like work, agree with router",
            "final_category": "work",
            "confidence": 0.95,
            "risk_score": 0.05,
            "needs_review": False,
            "explanation": "confirmed",
        })]
    ])
    with patch.object(evaluator_mod.anthropic, "Anthropic", factory):
        result = evaluator_mod.evaluate(_msg(), state)

    assert result.final_category == Category.WORK
    assert result.confidence == 0.95
    assert result.risk_score == 0.05
    assert result.needs_review is False
    assert result.reasoning.startswith("looks like work")
    assert state.risk_scores["e1"] == 0.05


def test_evaluator_can_override_router_to_risk():
    state = _state_with_router(Category.BILLING, 0.71)
    factory = fake_anthropic_with_tool_calls([
        [("verify_classification", {
            "reasoning": "typosquat sender, urgency tactics — phishing",
            "final_category": "risk",
            "confidence": 0.93,
            "risk_score": 0.95,
            "needs_review": True,
            "explanation": "phishing",
        })]
    ])
    with patch.object(evaluator_mod.anthropic, "Anthropic", factory):
        result = evaluator_mod.evaluate(_msg(), state)
    assert result.final_category == Category.RISK
    assert result.needs_review is True
    assert state.classifications["e1"] == Category.RISK


def test_evaluator_threshold_forces_needs_review_when_risk_high():
    state = _state_with_router(Category.WORK, 0.9)
    factory = fake_anthropic_with_tool_calls([
        [("verify_classification", {
            "reasoning": "high risk content",
            "final_category": "work",
            "confidence": 0.9,
            "risk_score": 0.85,
            "needs_review": False,  # model said false but threshold should override
            "explanation": "x",
        })]
    ])
    with patch.object(evaluator_mod.anthropic, "Anthropic", factory):
        result = evaluator_mod.evaluate(_msg(), state)
    assert result.needs_review is True


def test_evaluator_falls_back_on_api_error():
    state = _state_with_router(Category.WORK, 0.9)
    factory = fake_anthropic_with_tool_calls([None])
    with patch.object(evaluator_mod.anthropic, "Anthropic", factory):
        result = evaluator_mod.evaluate(_msg(), state)
    assert result.needs_review is True
    assert "evaluator failed" in result.explanation
