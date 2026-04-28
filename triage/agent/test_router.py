from unittest.mock import patch

from triage.agent import router as router_mod
from triage.agent._test_helpers import fake_anthropic_with_tool_calls
from triage.schemas import Category, Message, State


def _msg() -> Message:
    return Message(id="e1", subject="hi", sender="a@b", body_plain="...")


def test_route_writes_classification_and_confidence():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([
        [("classify_email", {"category": "work", "confidence": 0.92, "explanation": "ok"})]
    ])
    with patch.object(router_mod.anthropic, "Anthropic", factory):
        router_mod.route(_msg(), state)

    assert state.classifications["e1"] == Category.WORK
    assert state.confidence_scores["e1"] == 0.92


def test_route_clamps_out_of_range_confidence():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([
        [("classify_email", {"category": "personal", "confidence": 1.5, "explanation": "x"})]
    ])
    with patch.object(router_mod.anthropic, "Anthropic", factory):
        router_mod.route(_msg(), state)
    assert state.confidence_scores["e1"] == 1.0


def test_route_falls_back_to_unclassified_on_api_error():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([None])
    with patch.object(router_mod.anthropic, "Anthropic", factory):
        router_mod.route(_msg(), state)
    assert state.classifications["e1"] == Category.UNCLASSIFIED
    assert state.confidence_scores["e1"] == 0.0


def test_route_passes_feedback_when_supplied():
    state = State(messages=[_msg()])
    seen_messages: list = []

    factory = fake_anthropic_with_tool_calls([
        [("classify_email", {"category": "personal", "confidence": 0.91, "explanation": "ok"})]
    ])
    original_create = None
    captured = {}
    with patch.object(router_mod.anthropic, "Anthropic") as anth:
        from unittest.mock import MagicMock
        client = MagicMock()
        block = MagicMock()
        block.type = "tool_use"
        block.name = "classify_email"
        block.input = {"category": "personal", "confidence": 0.91, "explanation": "ok"}
        resp = MagicMock()
        resp.content = [block]
        def capture(**kwargs):
            captured.update(kwargs)
            return resp
        client.messages.create.side_effect = capture
        anth.return_value = client

        router_mod.route(_msg(), state, feedback="too low confidence")

    feedback_messages = [m for m in captured["messages"] if "Evaluator" in (
        m["content"][0]["text"] if isinstance(m.get("content"), list) else m.get("content", "")
    )]
    assert feedback_messages, "feedback message should be injected before final email"
