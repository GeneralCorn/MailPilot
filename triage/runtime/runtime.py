"""Runtime: tool execution + trace recording."""
from __future__ import annotations

from triage.schemas import ToolCall, ToolResult
from triage.tools import execute_tool_call

from .state import RuntimeState
from .trace import TraceEvent, TraceRecord


class Runtime:
    """Wraps tool execution so each ToolCall is recorded in a TraceRecord."""

    def __init__(
        self,
        *,
        state: RuntimeState | None = None,
        trace: TraceRecord | None = None,
    ) -> None:
        self._state = state or RuntimeState()
        self._trace = trace or TraceRecord()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def trace(self) -> TraceRecord:
        return self._trace

    def run_tool(self, tool_call: ToolCall) -> ToolResult:
        self._trace.append("runtime", "tool_call", tool=tool_call.tool.value, reason=tool_call.reason)
        result = execute_tool_call(tool_call)
        self._trace.append(
            "runtime", "tool_result",
            tool=result.tool.value,
            success=result.success,
            message=result.message,
        )
        return result

    def trace_events(self) -> list[TraceEvent]:
        return self._trace.events()

    def trace_to_list(self) -> list[dict]:
        return self._trace.to_list()
