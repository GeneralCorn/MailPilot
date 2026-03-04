from datetime import datetime, timezone
from typing import Callable

from ..storage import _load, _save
from ..schemas import Action, Priority, Status, ToolCall, ToolResult


def _get_email(emails: list[dict], email_id: str) -> dict:
    for e in emails:
        if e.get("id") == email_id:
            return e
    raise KeyError(f"Email {email_id!r} not found")


def _ok(tool: Action, message: str = "", data: dict = {}) -> ToolResult:
    return ToolResult(tool=tool, success=True, message=message, data=data)


def label_email(email_id: str, category: str) -> ToolResult:
    """Set or change email["category"] (e.g. work, personal, billing, risk, marketing)."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["category"] = category
    _save(emails)
    return _ok(Action.LABEL, f"Labelled as {category!r}", {"category": category})


def flag_email(email_id: str, flag: bool = True) -> ToolResult:
    """Mark an email as important / needing attention."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["flagged"] = flag
    _save(emails)
    return _ok(Action.FLAG, f"Flagged={flag}", {"flagged": flag})


def archive_email(email_id: str, folder: str = "archive") -> ToolResult:
    """Move email to a logical folder, removing it from the main inbox view."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["folder"] = folder
    email["status"] = Status.DONE.value
    _save(emails)
    return _ok(Action.ARCHIVE, f"Moved to {folder!r}", {"folder": folder})


def draft_reply(email_id: str, body: str, subject_override: str = "") -> ToolResult:
    """Attach an LLM-generated reply draft for human review (no sending)."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["draft_reply"] = {
        "body": body,
        "subject": subject_override or email.get("subject", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(emails)
    return _ok(Action.REPLY_DRAFT, "Draft saved", email["draft_reply"])


def create_calendar_event(
    email_id: str,
    title: str,
    start_time: str,
    end_time: str,
    location: str = "",
    response: str = "accept",
) -> ToolResult:
    """Store a calendar event derived from the email."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["calendar_event"] = {
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "response": response,
    }
    email["status"] = Status.DONE.value
    _save(emails)
    return _ok(Action.CALENDAR, f"Event '{title}' recorded", email["calendar_event"])


def escalate_email(email_id: str, target: str, reason: str) -> ToolResult:
    """Route a risky/ambiguous email to a human reviewer."""
    emails = _load()
    email = _get_email(emails, email_id)
    email["status"] = Status.NEEDS_REVIEW.value
    email.setdefault("escalations", []).append({
        "target": target,
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    _save(emails)
    return _ok(Action.ESCALATE, f"Escalated to {target!r}", {"reason": reason})


def no_action(email_id: str, reason: str = "") -> ToolResult:
    """Explicitly record that no action is needed."""
    return _ok(Action.NO_ACTION, reason or "No action required", {"email_id": email_id})



def set_status(email_id: str, status: Status) -> ToolResult:
    emails = _load()
    email = _get_email(emails, email_id)
    email["status"] = status.value
    _save(emails)
    return _ok(Action.NO_ACTION, f"Status → {status.value}", {"status": status.value})


def update_priority(email_id: str, priority: Priority, priority_rank: int) -> ToolResult:
    emails = _load()
    email = _get_email(emails, email_id)
    email["priority"] = priority.value
    email["priority_rank"] = priority_rank
    _save(emails)
    return _ok(Action.NO_ACTION, f"Priority → {priority.value} (rank {priority_rank})")


def add_note(email_id: str, note: str, source: str = "") -> ToolResult:
    emails = _load()
    email = _get_email(emails, email_id)
    email.setdefault("notes", []).append({
        "text": note,
        "source": source,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    _save(emails)
    return _ok(Action.NO_ACTION, "Note added", {"note": note, "source": source})


def record_tool_result(tool_call: ToolCall, success: bool, message: str, data: dict = {}) -> ToolResult:
    """Build a ToolResult from a ToolCall; use this to centralise trace logging."""
    return ToolResult(tool=tool_call.tool, success=success, message=message, data=data)



def batch_label(email_ids: list[str], category: str) -> list[ToolResult]:
    return [label_email(eid, category) for eid in email_ids]


def batch_archive(email_ids: list[str], folder: str = "archive") -> list[ToolResult]:
    return [archive_email(eid, folder) for eid in email_ids]



TOOLS = [
    label_email,
    flag_email,
    archive_email,
    draft_reply,
    create_calendar_event,
    escalate_email,
    no_action,
    set_status,
    update_priority,
    add_note,
    record_tool_result,
    batch_label,
    batch_archive,
]


_TOOL_DISPATCH: dict[Action, Callable[..., ToolResult]] = {
    Action.LABEL: label_email,
    Action.FLAG: flag_email,
    Action.ARCHIVE: archive_email,
    Action.REPLY_DRAFT: draft_reply,
    Action.CALENDAR: create_calendar_event,
    Action.ESCALATE: escalate_email,
    Action.NO_ACTION: no_action,
}


def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    """Dispatch a ToolCall to the appropriate tool function and return a ToolResult."""
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

def apply_tool_calls(state: State) -> State:
    """Execute all pending ToolCalls in a State and append their results."""
    if not state.pending_calls:
        return state

    new_results: list[ToolResult] = []
    for call in list(state.pending_calls):
        result = execute_tool_call(call)
        new_results.append(result)

    state.results.extend(new_results)
    state.pending_calls.clear()
    return state