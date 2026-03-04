from datetime import datetime, timezone
from typing import Tuple

from .schemas import State, Trace, ToolCall, ToolResult
from .tools.tool_registry import execute_tool_call


def run_tool_calls(state: State, trace: Trace | None = None) -> Tuple[State, Trace]:
    """
    Execute all pending ToolCalls in a State, append their ToolResults,
    and keep a Trace of what happened.
    """
    if trace is None:
        trace = Trace(
            run_id=f"run-{datetime.now(timezone.utc).isoformat()}",
            started_at=datetime.now(timezone.utc),
            input_count=len(state.messages),
        )

    if not state.pending_calls:
        trace.finished_at = datetime.now(timezone.utc)
        return state, trace

    for call in list(state.pending_calls):
        trace.tool_calls.append(call)
        result: ToolResult = execute_tool_call(call)
        state.results.append(result)
        trace.tool_results.append(result)

    state.pending_calls.clear()
    trace.finished_at = datetime.now(timezone.utc)
    return state, trace

