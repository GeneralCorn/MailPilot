"""Shared test helpers for mocking the Anthropic tool-use API."""
from __future__ import annotations

from unittest.mock import MagicMock

import anthropic


def make_tool_use_block(name: str, input_dict: dict):
    """Mock a single tool_use content block as returned by anthropic SDK."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = dict(input_dict)
    return block


def fake_anthropic_with_tool_calls(calls_per_request: list[list[tuple[str, dict]]]):
    """Build a callable that fakes anthropic.Anthropic.

    `calls_per_request` is a list, one entry per expected create() call. Each entry
    is a list of (tool_name, input_dict) pairs becoming tool_use blocks.
    To simulate API errors, put None at that position instead of a list.
    """
    iterator = iter(calls_per_request)

    def factory(*args, **kwargs):
        client = MagicMock()
        def create(**_):
            entry = next(iterator)
            if entry is None:
                raise anthropic.APIError(message="boom", request=MagicMock(), body=None)
            resp = MagicMock()
            resp.content = [make_tool_use_block(name, inp) for name, inp in entry]
            resp.usage.input_tokens = 10
            resp.usage.output_tokens = 5
            return resp
        client.messages.create.side_effect = lambda **kw: create(**kw)
        return client

    return factory
