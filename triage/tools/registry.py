from typing import Callable

from ..schemas import Action, ToolCall, ToolResult
from .actions import (
    archive_email,
    create_calendar_event,
    draft_reply,
    escalate_email,
    flag_email,
    label_email,
    no_action,
    send_confirmation_email,
    send_rsvp_email,
    summarize_email,
)


def _send_email_dispatch(email_id: str, kind: str, body: str, **extra) -> ToolResult:
    if kind == "rsvp":
        return send_rsvp_email(email_id, decision=extra.get("decision", "accept"), body=body)
    if kind == "confirmation":
        return send_confirmation_email(email_id, body=body, subject_override=extra.get("subject_override", ""))
    return ToolResult(
        tool=Action.SEND_EMAIL,
        success=False,
        message=f"unknown send_email kind: {kind!r}",
        data={"kind": kind},
    )


_TOOL_DISPATCH: dict[Action, Callable[..., ToolResult]] = {
    Action.LABEL: label_email,
    Action.FLAG: flag_email,
    Action.ARCHIVE: archive_email,
    Action.REPLY_DRAFT: draft_reply,
    Action.CALENDAR: create_calendar_event,
    Action.ESCALATE: escalate_email,
    Action.SUMMARIZE: summarize_email,
    Action.SEND_EMAIL: _send_email_dispatch,
    Action.NO_ACTION: no_action,
}


def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    """Dispatch a ToolCall to the appropriate action and return a ToolResult."""
    func = _TOOL_DISPATCH.get(tool_call.tool)
    if func is None:
        return ToolResult(
            tool=tool_call.tool,
            success=False,
            message=f"Unknown tool: {tool_call.tool!s}",
            data={"tool": str(tool_call.tool)},
        )

    try:
        result = func(**tool_call.parameters)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(
            tool=tool_call.tool,
            success=True,
            message=str(result),
            data={"result": result},
        )
    except Exception as exc:
        return ToolResult(
            tool=tool_call.tool,
            success=False,
            message=str(exc),
            data={"error_type": type(exc).__name__},
        )
