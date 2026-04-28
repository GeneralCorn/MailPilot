from triage.agent._caching import to_cached_request
from triage.schemas import AgentMessage


def test_to_cached_request_marks_last_assistant_block():
    msgs = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="fewshot u1"),
        AgentMessage(role="assistant", content="fewshot a1"),
        AgentMessage(role="user", content="fewshot u2"),
        AgentMessage(role="assistant", content="fewshot a2"),
        AgentMessage(role="user", content="actual email"),
    ]

    system, out = to_cached_request(msgs)

    assert system == "sys"
    assert len(out) == 5
    # only the last assistant block is tagged
    last_assistant = next(m for m in out if m["role"] == "assistant" and m["content"][0]["text"] == "fewshot a2")
    assert last_assistant["content"][0].get("cache_control") == {"type": "ephemeral"}

    # earlier assistant block is not tagged
    earlier_assistant = next(m for m in out if m["role"] == "assistant" and m["content"][0]["text"] == "fewshot a1")
    assert "cache_control" not in earlier_assistant["content"][0]

    # user blocks are never tagged
    for m in out:
        if m["role"] == "user":
            assert "cache_control" not in m["content"][0]


def test_to_cached_request_handles_no_assistant_messages():
    msgs = [
        AgentMessage(role="system", content="sys"),
        AgentMessage(role="user", content="just a question"),
    ]
    system, out = to_cached_request(msgs)
    assert system == "sys"
    assert len(out) == 1
    assert "cache_control" not in out[0]["content"][0]


def test_to_cached_request_preserves_message_order():
    msgs = [
        AgentMessage(role="user", content="a"),
        AgentMessage(role="assistant", content="b"),
        AgentMessage(role="user", content="c"),
    ]
    _, out = to_cached_request(msgs)
    assert [m["content"][0]["text"] for m in out] == ["a", "b", "c"]
