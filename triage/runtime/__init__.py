"""
MailPilot Runtime
=================
System runtime: message routing, state management, tool execution, trace.
"""
from .runtime import Runtime
from .message import DispatchMessage
from .state import RuntimeState
from .trace import TraceRecord, TraceEvent
from . import tools
from . import agents

__all__ = [
    "Runtime",
    "DispatchMessage",
    "RuntimeState",
    "TraceRecord",
    "TraceEvent",
    "create_mailpilot_runtime",
]


def create_mailpilot_runtime(
    *,
    register_tools: bool = True,
    register_agents: bool = True,
) -> Runtime:
    """Create a Runtime with optional default tools and triage agents (router, evaluator, ranker, worker)."""
    rt = Runtime()
    if register_tools:
        tools.register_default_tools()
    if register_agents:
        agents.register_triage_agents(rt)
    return rt
