from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from triage.agent import ranker as ranker_mod
from triage.agent._test_helpers import fake_anthropic_with_tool_calls
from triage.schemas import Category, Message, Priority, State


def _msg(eid: str, subject: str = "x", received: str | None = None) -> Message:
    rec = datetime.fromisoformat(received) if received else None
    return Message(id=eid, subject=subject, sender="a@b", body_plain="...", received_at=rec)


def test_rank_writes_priority_queue_sorted_desc():
    state = State(messages=[_msg("a"), _msg("b"), _msg("c")])
    state.classifications.update({"a": Category.WORK, "b": Category.MARKETING, "c": Category.BILLING})

    factory = fake_anthropic_with_tool_calls([
        [("rank_emails", {
            "ranked": [
                {"email_id": "a", "score": 0.9, "priority": "urgent", "reason": "x"},
                {"email_id": "b", "score": 0.1, "priority": "minimal", "reason": "x"},
                {"email_id": "c", "score": 0.5, "priority": "normal", "reason": "x"},
            ]
        })]
    ])
    with patch.object(ranker_mod.anthropic, "Anthropic", factory):
        ranker_mod.rank(state)

    assert state.priority_queue == [("a", 0.9), ("c", 0.5), ("b", 0.1)]
    assert state.priorities["a"] == Priority.URGENT
    assert state.priorities["c"] == Priority.NORMAL
    assert state.priorities["b"] == Priority.MINIMAL


def test_rank_falls_back_on_api_error():
    state = State(messages=[
        _msg("a", received="2026-04-01T00:00:00+00:00"),
        _msg("b", received="2026-04-03T00:00:00+00:00"),
        _msg("c", received="2026-04-02T00:00:00+00:00"),
    ])
    factory = fake_anthropic_with_tool_calls([None])
    with patch.object(ranker_mod.anthropic, "Anthropic", factory):
        ranker_mod.rank(state)

    assert [eid for eid, _ in state.priority_queue] == ["b", "c", "a"]
    assert all(state.priorities[eid] == Priority.NORMAL for eid in ("a", "b", "c"))
    assert set(state.needs_review) == {"a", "b", "c"}


def test_rank_falls_back_on_malformed_tool_input():
    state = State(messages=[_msg("a"), _msg("b")])
    # missing required key -> handler treats as fallback
    factory = fake_anthropic_with_tool_calls([
        [("rank_emails", {"ranked": [{"email_id": "a"}]})]  # missing score / priority / reason
    ])
    with patch.object(ranker_mod.anthropic, "Anthropic", factory):
        ranker_mod.rank(state)
    # fallback assigns NORMAL to all and pushes to needs_review
    assert state.priorities["a"] == Priority.NORMAL


def test_rank_clamps_score_out_of_range():
    state = State(messages=[_msg("a"), _msg("b")])
    factory = fake_anthropic_with_tool_calls([
        [("rank_emails", {
            "ranked": [
                {"email_id": "a", "score": 1.5, "priority": "urgent", "reason": "x"},
                {"email_id": "b", "score": -0.2, "priority": "minimal", "reason": "x"},
            ]
        })]
    ])
    with patch.object(ranker_mod.anthropic, "Anthropic", factory):
        ranker_mod.rank(state)
    scores = dict(state.priority_queue)
    assert scores["a"] == 1.0
    assert scores["b"] == 0.0


def test_rank_caps_at_max_batch_and_appends_overflow_tail():
    msgs = [_msg(f"e{i}") for i in range(ranker_mod.MAX_BATCH + 5)]
    state = State(messages=msgs)
    ranked = [
        {"email_id": f"e{i}", "score": 0.5, "priority": "normal", "reason": "x"}
        for i in range(ranker_mod.MAX_BATCH)
    ]
    factory = fake_anthropic_with_tool_calls([
        [("rank_emails", {"ranked": ranked})]
    ])
    with patch.object(ranker_mod.anthropic, "Anthropic", factory):
        ranker_mod.rank(state)

    tail = state.priority_queue[-5:]
    assert [eid for eid, _ in tail] == [f"e{i}" for i in range(ranker_mod.MAX_BATCH, ranker_mod.MAX_BATCH + 5)]
    assert all(score == 0.0 for _, score in tail)
    for i in range(ranker_mod.MAX_BATCH, ranker_mod.MAX_BATCH + 5):
        assert state.priorities[f"e{i}"] == Priority.NORMAL


def test_rank_handles_empty_state_as_noop():
    state = State()
    ranker_mod.rank(state)
    assert state.priority_queue == []
    assert state.priorities == {}


def test_rank_short_circuits_for_single_email_without_llm():
    state = State(messages=[_msg("solo")])

    def boom(*args, **kwargs):
        raise AssertionError("anthropic.Anthropic should not be called for single-email rank")

    with patch.object(ranker_mod.anthropic, "Anthropic", boom):
        ranker_mod.rank(state)

    assert state.priorities["solo"] == Priority.NORMAL
    assert state.priority_queue == [("solo", 0.5)]
