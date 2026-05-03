from unittest.mock import MagicMock, patch

import anthropic
import pytest

from triage.agent._anthropic import call_tools
from triage.agent._test_helpers import fake_anthropic_with_tool_calls


def test_call_tools_returns_parsed_tool_use_blocks():
    factory = fake_anthropic_with_tool_calls([
        [("classify_email", {"category": "work", "confidence": 0.9, "explanation": "ok"})]
    ])
    with patch("anthropic.Anthropic", factory):
        client = anthropic.Anthropic(api_key="test")
        result = call_tools(
            client, model="m", system="s", messages=[], tools=[{"name": "classify_email"}],
            force_tool="classify_email",
        )
    assert result == [{"name": "classify_email", "input": {"category": "work", "confidence": 0.9, "explanation": "ok"}}]


def test_call_tools_supports_multiple_tool_uses_in_one_response():
    factory = fake_anthropic_with_tool_calls([
        [("summarize", {"summary": "x"}), ("archive", {"folder": "promotions"})]
    ])
    with patch("anthropic.Anthropic", factory):
        client = anthropic.Anthropic(api_key="test")
        result = call_tools(
            client, model="m", system="s", messages=[], tools=[],
        )
    assert [r["name"] for r in result] == ["summarize", "archive"]
    assert result[1]["input"] == {"folder": "promotions"}


def test_call_tools_returns_none_on_api_error():
    factory = fake_anthropic_with_tool_calls([None])
    with patch("anthropic.Anthropic", factory):
        client = anthropic.Anthropic(api_key="test")
        result = call_tools(client, model="m", system="s", messages=[], tools=[])
    assert result is None


def test_call_tools_skips_non_tool_use_blocks():
    """If the model returns a text block alongside tool_use, ignore the text."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "some chatter"

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "classify_email"
    tool_block.input = {"category": "work"}

    fake_resp = MagicMock()
    fake_resp.content = [text_block, tool_block]
    fake_resp.usage.input_tokens = 0
    fake_resp.usage.output_tokens = 0

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    result = call_tools(fake_client, model="m", system="s", messages=[], tools=[])
    assert result == [{"name": "classify_email", "input": {"category": "work"}}]
