from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic

from triage.schemas import Message

from benchmarks.baselines import single_prompt


def _msg(eid: str) -> Message:
    return Message(id=eid, subject="hi", sender="x@y.com", body_plain="...")


def _fake_client(rows_per_email: list[dict | None]):
    """Mock anthropic.Anthropic so each create() returns the next row in the list."""
    iterator = iter(rows_per_email)

    def factory(*args, **kwargs):
        client = MagicMock()
        def create(**_):
            row = next(iterator)
            if row is None:
                raise anthropic.APIError(message="boom", request=MagicMock(), body=None)
            block = MagicMock()
            block.type = "tool_use"
            block.name = "submit"
            block.input = row
            resp = MagicMock()
            resp.content = [block]
            resp.usage.input_tokens = 5
            resp.usage.output_tokens = 3
            return resp
        client.messages.create.side_effect = lambda **kw: create(**kw)
        return client

    return factory


def test_run_uses_per_email_response():
    factory = _fake_client([
        {"category": "work", "priority": "high", "action": "flag", "needs_review": False},
        {"category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
    ])
    with patch("anthropic.Anthropic", factory):
        out = single_prompt.run([_msg("a"), _msg("b")])

    assert out[0]["category"] == "work"
    assert out[1]["category"] == "marketing"
    assert all(r["proposed_actions"] == [] for r in out)


def test_run_falls_back_on_api_error():
    factory = _fake_client([None])
    with patch("anthropic.Anthropic", factory):
        out = single_prompt.run([_msg("a")])
    assert out[0]["category"] == "marketing"
    assert out[0]["action"] == "no_action"
