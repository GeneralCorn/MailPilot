from __future__ import annotations

# Module-level accumulator. Pipelines run serially, so a global is enough.
# Reset before a run, snapshot after.
_calls = 0
_input_tokens = 0
_output_tokens = 0


def reset() -> None:
    global _calls, _input_tokens, _output_tokens
    _calls = 0
    _input_tokens = 0
    _output_tokens = 0


def record(input_tokens: int, output_tokens: int) -> None:
    global _calls, _input_tokens, _output_tokens
    _calls += 1
    _input_tokens += input_tokens
    _output_tokens += output_tokens


def snapshot() -> dict:
    return {
        "calls": _calls,
        "input_tokens": _input_tokens,
        "output_tokens": _output_tokens,
    }
