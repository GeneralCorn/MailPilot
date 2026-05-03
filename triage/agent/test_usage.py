from __future__ import annotations

from unittest.mock import patch

import anthropic

from triage.agent import _usage
from triage.agent._anthropic import call_tools
from triage.agent._test_helpers import fake_anthropic_with_tool_calls


def test_call_tools_records_usage_per_call():
    _usage.reset()
    factory = fake_anthropic_with_tool_calls([
        [("foo", {})],
        [("bar", {})],
    ])
    with patch("anthropic.Anthropic", factory):
        client = anthropic.Anthropic(api_key="t")
        call_tools(client, model="m", system="s", messages=[], tools=[])
        call_tools(client, model="m", system="s", messages=[], tools=[])

    snap = _usage.snapshot()
    # _test_helpers seeds usage as 10 in / 5 out per call
    assert snap == {"calls": 2, "input_tokens": 20, "output_tokens": 10}


def test_reset_clears_accumulator():
    _usage.reset()
    factory = fake_anthropic_with_tool_calls([[("foo", {})]])
    with patch("anthropic.Anthropic", factory):
        call_tools(anthropic.Anthropic(api_key="t"), model="m", system="s", messages=[], tools=[])
    _usage.reset()
    assert _usage.snapshot() == {"calls": 0, "input_tokens": 0, "output_tokens": 0}
