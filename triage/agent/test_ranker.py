from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from triage.agent import ranker as ranker_mod
from triage.schemas import Category, Message, Priority, State


def _msg(eid: str, subject: str = "x", received: str | None = None) -> Message:
    rec = datetime.fromisoformat(received) if received else None
    return Message(id=eid, subject=subject, sender="a@b", body_plain="...", received_at=rec)


def _fake_anthropic_factory(texts: list[str]):
    """Return a callable that builds a fake anthropic.Anthropic client returning `texts` in order."""
    iterator = iter(texts)

    def factory(*args, **kwargs):
        client = MagicMock()
        def create(**_):
            text = next(iterator)
            resp = MagicMock()
            resp.content = [MagicMock(text=text)]
            return resp
        client.messages.create.side_effect = lambda **kw: create(**kw)
        return client

    return factory


def test_rank_writes_priority_queue_sorted_desc():
    state = State(messages=[_msg("a"), _msg("b"), _msg("c")])
    state.classifications.update({"a": Category.WORK, "b": Category.MARKETING, "c": Category.BILLING})

    response_json = (
        '{"ranked":['
        '{"email_id":"a","score":0.9,"priority":"urgent","reason":"x"},'
        '{"email_id":"b","score":0.1,"priority":"minimal","reason":"x"},'
        '{"email_id":"c","score":0.5,"priority":"normal","reason":"x"}'
        "]}"
    )

    with patch.object(ranker_mod.anthropic, "Anthropic", _fake_anthropic_factory([response_json])):
        ranker_mod.rank(state)

    assert state.priority_queue == [("a", 0.9), ("c", 0.5), ("b", 0.1)]
    assert state.priorities["a"] == Priority.URGENT
    assert state.priorities["c"] == Priority.NORMAL
    assert state.priorities["b"] == Priority.MINIMAL


def test_rank_repair_retry_succeeds_on_second_attempt():
    state = State(messages=[_msg("a")])
    state.classifications["a"] = Category.WORK

    bad = "not json at all"
    good = '{"ranked":[{"email_id":"a","score":0.7,"priority":"important","reason":"x"}]}'

    with patch.object(ranker_mod.anthropic, "Anthropic", _fake_anthropic_factory([bad, good])):
        ranker_mod.rank(state)

    assert state.priorities["a"] == Priority.IMPORTANT
    assert state.priority_queue == [("a", 0.7)]
    assert "a" not in state.needs_review


def test_rank_falls_back_on_persistent_failure():
    state = State(messages=[_msg("a", received="2026-04-01T00:00:00+00:00"),
                            _msg("b", received="2026-04-03T00:00:00+00:00"),
                            _msg("c", received="2026-04-02T00:00:00+00:00")])

    with patch.object(ranker_mod.anthropic, "Anthropic", _fake_anthropic_factory(["bad", "still bad"])):
        ranker_mod.rank(state)

    # received_at desc: b, c, a
    assert [eid for eid, _ in state.priority_queue] == ["b", "c", "a"]
    assert all(state.priorities[eid] == Priority.NORMAL for eid in ("a", "b", "c"))
    assert set(state.needs_review) == {"a", "b", "c"}


def test_rank_clamps_score_out_of_range():
    state = State(messages=[_msg("a"), _msg("b")])
    response = (
        '{"ranked":['
        '{"email_id":"a","score":1.5,"priority":"urgent","reason":"x"},'
        '{"email_id":"b","score":-0.2,"priority":"minimal","reason":"x"}'
        "]}"
    )
    with patch.object(ranker_mod.anthropic, "Anthropic", _fake_anthropic_factory([response])):
        ranker_mod.rank(state)
    scores = dict(state.priority_queue)
    assert scores["a"] == 1.0
    assert scores["b"] == 0.0


def test_rank_caps_at_max_batch_and_appends_overflow_tail():
    msgs = [_msg(f"e{i}") for i in range(ranker_mod.MAX_BATCH + 5)]
    state = State(messages=msgs)
    # build a ranked entry only for the first MAX_BATCH ids, all score 0.5
    ranked_entries = ",".join(
        f'{{"email_id":"e{i}","score":0.5,"priority":"normal","reason":"x"}}'
        for i in range(ranker_mod.MAX_BATCH)
    )
    response = '{"ranked":[' + ranked_entries + "]}"

    with patch.object(ranker_mod.anthropic, "Anthropic", _fake_anthropic_factory([response])):
        ranker_mod.rank(state)

    # overflow ids show up at the end with score 0.0, in arrival order
    tail = state.priority_queue[-5:]
    assert [eid for eid, _ in tail] == [f"e{i}" for i in range(ranker_mod.MAX_BATCH, ranker_mod.MAX_BATCH + 5)]
    assert all(score == 0.0 for _, score in tail)
    # overflow priorities default to NORMAL
    for i in range(ranker_mod.MAX_BATCH, ranker_mod.MAX_BATCH + 5):
        assert state.priorities[f"e{i}"] == Priority.NORMAL


def test_rank_handles_empty_state_as_noop():
    state = State()
    ranker_mod.rank(state)
    assert state.priority_queue == []
    assert state.priorities == {}


def test_rank_falls_back_on_anthropic_api_error():
    state = State(messages=[_msg("a"), _msg("b")])

    def boom_factory(*args, **kwargs):
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIError(
            message="boom", request=MagicMock(), body=None
        )
        return client

    with patch.object(ranker_mod.anthropic, "Anthropic", boom_factory):
        ranker_mod.rank(state)

    assert {eid for eid, _ in state.priority_queue} == {"a", "b"}
    assert all(state.priorities[eid] == Priority.NORMAL for eid in ("a", "b"))
    assert set(state.needs_review) == {"a", "b"}
