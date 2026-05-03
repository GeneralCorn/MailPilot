from __future__ import annotations

from triage.schemas import Category, Message, State

from benchmarks.baselines.no_evaluator import passthrough_evaluator


def test_passthrough_returns_router_category_as_reliable():
    state = State(messages=[])
    state.classifications["e1"] = Category.WORK
    msg = Message(id="e1", subject="x", sender="a@b", body_plain="...")
    r = passthrough_evaluator(msg, state)
    assert r.final_category == Category.WORK
    assert r.confidence == 1.0
    assert r.risk_score == 0.0
    assert r.needs_review is False


def test_passthrough_defaults_to_unclassified_when_router_silent():
    state = State(messages=[])
    msg = Message(id="x", subject="x", sender="a@b", body_plain="...")
    r = passthrough_evaluator(msg, state)
    assert r.final_category == Category.UNCLASSIFIED
