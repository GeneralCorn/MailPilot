"""
Runtime: message routing, state management, tool execution, trace.
"""
from __future__ import annotations
from typing import Any, Callable

from triage.schemas import ToolCall, ToolResult

from .message import DispatchMessage
from .state import RuntimeState
from .trace import TraceRecord, TraceEvent
from . import tools as tool_module

# Agent handler: (payload: dict, state: RuntimeState) -> Any
AgentHandler = Callable[[dict[str, Any], RuntimeState], Any]


class Runtime:
    """
    System runtime: routes messages to agents, holds state, runs tools, records trace.
    """

    def __init__(
        self,
        *,
        state: RuntimeState | None = None,
        trace: TraceRecord | None = None,
    ) -> None:
        self._state = state or RuntimeState()
        self._trace = trace or TraceRecord()
        self._agents: dict[str, AgentHandler] = {}

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def trace(self) -> TraceRecord:
        return self._trace

    def register_agent(self, name: str, handler: AgentHandler) -> None:
        """Register an agent that can handle dispatched messages."""
        self._agents[name] = handler

    def dispatch(self, message: DispatchMessage) -> Any:
        """
        Route a message to the correct agent.
        Records agent_start / agent_end in trace.
        """
        target = message.target
        payload = message.payload

        self._trace.append(target, "agent_start", payload_keys=list(payload.keys()))
        if target not in self._agents:
            self._trace.append("runtime", "error", target_agent=target, reason="unknown_agent")
            raise ValueError(f"Unknown agent: {target}")

        handler = self._agents[target]
        try:
            out = handler(payload, self._state)
            self._trace.append(target, "agent_end", payload_keys=list(payload.keys()))
            return out
        except Exception as e:
            self._trace.append("runtime", "error", target_agent=target, error=str(e))
            raise

    def run_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call via the tool registry."""
        self._trace.append("runtime", "tool_call", tool=tool_call.tool.value, reason=tool_call.reason)
        result = tool_module.run_tool(tool_call)
        self._trace.append(
            "runtime", "tool_result",
            tool=result.tool.value,
            success=result.success,
            message=result.message,
        )
        return result

    def run_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in order."""
        return [self.run_tool(tc) for tc in tool_calls]

    def trace_events(self) -> list[TraceEvent]:
        """Return all trace events (read-only)."""
        return self._trace.events()

    def trace_to_list(self) -> list[dict[str, Any]]:
        """Return trace as list of dicts for logging/serialization."""
        return self._trace.to_list()
