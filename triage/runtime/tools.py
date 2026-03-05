"""Tool registry and runner for ToolCall execution."""
from __future__ import annotations
from typing import Any, Callable
from triage.schemas import Action, ToolCall, ToolResult

# (parameters dict) -> result dict or None
ToolFn = Callable[..., Any]
TOOL_REGISTRY: dict[Action, ToolFn] = {}


def _stub_label(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "label applied", "data": kwargs}


def _stub_flag(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "flagged", "data": kwargs}


def _stub_archive(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "archived", "data": kwargs}


def _stub_reply_draft(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "draft created", "data": kwargs}


def _stub_calendar(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "calendar event created", "data": kwargs}


def _stub_escalate(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "escalated", "data": kwargs}


def _stub_no_action(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "message": "no action", "data": {}}


def register_default_tools() -> None:
    """Register stub implementations for all Action types. Replace with real impls as needed."""
    TOOL_REGISTRY[Action.LABEL] = _stub_label
    TOOL_REGISTRY[Action.FLAG] = _stub_flag
    TOOL_REGISTRY[Action.ARCHIVE] = _stub_archive
    TOOL_REGISTRY[Action.REPLY_DRAFT] = _stub_reply_draft
    TOOL_REGISTRY[Action.CALENDAR] = _stub_calendar
    TOOL_REGISTRY[Action.ESCALATE] = _stub_escalate
    TOOL_REGISTRY[Action.NO_ACTION] = _stub_no_action


def register_tool(action: Action, fn: ToolFn) -> None:
    """Register a handler for an action."""
    TOOL_REGISTRY[action] = fn


def run_tool(tool_call: ToolCall) -> ToolResult:
    """Execute a single tool call and return a ToolResult."""
    action = tool_call.tool
    params = tool_call.parameters or {}
    if action not in TOOL_REGISTRY:
        return ToolResult(
            tool=action,
            success=False,
            message=f"Unknown tool: {action}",
            data={},
        )
    try:
        result = TOOL_REGISTRY[action](**params)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, dict):
            return ToolResult(
                tool=action,
                success=result.get("success", True),
                message=result.get("message", ""),
                data=result.get("data", result),
            )
        return ToolResult(tool=action, success=True, message="", data={"result": result})
    except Exception as e:
        return ToolResult(
            tool=action,
            success=False,
            message=str(e),
            data={},
        )


def run_tools(tool_calls: list[ToolCall]) -> list[ToolResult]:
    """Execute multiple tool calls in order."""
    return [run_tool(tc) for tc in tool_calls]
