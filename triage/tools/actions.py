from datetime import datetime, timezone

from .. import persistence
from ..schemas import Action, Priority, Status, ToolResult


def _ok(tool: Action, message: str = "", data: dict | None = None) -> ToolResult:
    return ToolResult(tool=tool, success=True, message=message, data=data or {})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def label_email(email_id: str, category: str) -> ToolResult:
    persistence.update_email_state(email_id, category=category)
    return _ok(Action.LABEL, f"Labelled as {category!r}", {"category": category})


def flag_email(email_id: str, flag: bool = True) -> ToolResult:
    persistence.update_email_state(email_id, flagged=bool(flag))
    return _ok(Action.FLAG, f"Flagged={flag}", {"flagged": flag})


def archive_email(email_id: str, folder: str = "archive") -> ToolResult:
    persistence.update_email_state(email_id, folder=folder, status=Status.DONE.value)
    return _ok(Action.ARCHIVE, f"Moved to {folder!r}", {"folder": folder})


def draft_reply(email_id: str, body: str, subject_override: str = "") -> ToolResult:
    if not subject_override:
        from ..storage import get_email_body
        body_row = get_email_body(email_id) or {}
        subject_override = body_row.get("subject", "")
    payload = {"body": body, "subject": subject_override, "created_at": _utcnow()}
    persistence.update_email_state(email_id, draft_reply=payload)
    return _ok(Action.REPLY_DRAFT, "Draft saved", payload)


def create_calendar_event(
    email_id: str,
    title: str,
    start_time: str,
    end_time: str,
    location: str = "",
    response: str = "accept",
) -> ToolResult:
    event = {
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "response": response,
    }
    persistence.update_email_state(email_id, calendar_event=event, status=Status.DONE.value)
    return _ok(Action.CALENDAR, f"Event '{title}' recorded", event)


def escalate_email(email_id: str, target: str, reason: str) -> ToolResult:
    persistence.update_email_state(email_id, status=Status.FLAGGED.value)
    persistence.append_email_list_field(
        email_id, "escalations", {"target": target, "reason": reason, "at": _utcnow()}
    )
    return _ok(Action.ESCALATE, f"Escalated to {target!r}", {"reason": reason})


def no_action(email_id: str, reason: str = "") -> ToolResult:
    return _ok(Action.NO_ACTION, reason or "No action required", {"email_id": email_id})


def set_status(email_id: str, status: Status) -> ToolResult:
    persistence.update_email_state(email_id, status=status.value)
    return _ok(Action.NO_ACTION, f"Status → {status.value}", {"status": status.value})


def update_priority(email_id: str, priority: Priority, priority_rank: int) -> ToolResult:
    persistence.update_email_state(email_id, priority=priority.value, priority_rank=priority_rank)
    return _ok(Action.NO_ACTION, f"Priority → {priority.value} (rank {priority_rank})")


def add_note(email_id: str, note: str, source: str = "") -> ToolResult:
    persistence.append_email_list_field(
        email_id, "notes", {"text": note, "source": source, "at": _utcnow()}
    )
    return _ok(Action.NO_ACTION, "Note added", {"note": note, "source": source})


def batch_label(email_ids: list[str], category: str) -> list[ToolResult]:
    return [label_email(eid, category) for eid in email_ids]


def batch_archive(email_ids: list[str], folder: str = "archive") -> list[ToolResult]:
    return [archive_email(eid, folder) for eid in email_ids]
