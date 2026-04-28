import os
from datetime import datetime, timezone

from .. import persistence
from ..schemas import Action, Priority, Status, ToolResult


def _ok(tool: Action, message: str = "", data: dict | None = None) -> ToolResult:
    return ToolResult(tool=tool, success=True, message=message, data=data or {})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_mocked() -> bool:
    return os.environ.get("MAILPILOT_MOCK_TOOLS") == "1"


def _already_sent(email_id: str, kind: str) -> bool:
    for action in persistence.get_email_state(email_id).get("external_actions", []):
        if action.get("kind") == kind and action.get("success"):
            return True
    return False


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
    from ..storage import get_email_body
    body_row = get_email_body(email_id) or {}
    subject = subject_override or body_row.get("subject", "")
    payload = {"body": body, "subject": subject, "created_at": _utcnow()}

    if not _is_mocked():
        from email.mime.text import MIMEText
        import base64
        from .. import gmail
        svc = gmail.get_gmail_service()
        msg = MIMEText(body)
        msg["subject"] = subject
        if body_row.get("sender"):
            msg["to"] = body_row["sender"]
        raw_bytes = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = svc.users().drafts().create(
            userId="me", body={"message": {"raw": raw_bytes}}
        ).execute()
        payload["gmail_draft_id"] = draft.get("id")

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
    if not _is_mocked():
        from .. import gmail
        svc = gmail.get_calendar_service()
        body = {
            "summary": title,
            "location": location,
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }
        created = svc.events().insert(calendarId="primary", body=body).execute()
        event["calendar_event_id"] = created.get("id")
        event["html_link"] = created.get("htmlLink")

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


def summarize_email(email_id: str, summary: str) -> ToolResult:
    persistence.update_email_state(email_id, summary=summary)
    return _ok(Action.SUMMARIZE, "Summary saved", {"summary_chars": len(summary)})


def send_rsvp_email(email_id: str, decision: str, body: str) -> ToolResult:
    kind = f"rsvp:{decision}"
    if _already_sent(email_id, kind):
        return _ok(Action.SEND_EMAIL, "RSVP already sent", {"kind": "rsvp", "decision": decision, "skipped": True})

    from ..storage import get_email_body
    body_row = get_email_body(email_id) or {}
    to = body_row.get("sender", "")
    subject = f"Re: {body_row.get('subject', '')} — {decision.upper()}"

    if _is_mocked():
        sent = {"kind": kind, "to": to, "subject": subject, "success": True, "at": _utcnow(), "mock": True}
    else:
        from .. import gmail
        result = gmail.send_email(to, subject, body, thread_id=body_row.get("thread_id"))
        sent = {"kind": kind, "to": to, "subject": subject, "success": True,
                "at": _utcnow(), "gmail_message_id": result.get("id")}

    persistence.append_email_list_field(email_id, "external_actions", sent)
    return _ok(Action.SEND_EMAIL, f"RSVP {decision} sent to {to}",
               {"kind": "rsvp", "decision": decision, "to": to})


def send_confirmation_email(email_id: str, body: str, subject_override: str = "") -> ToolResult:
    if _already_sent(email_id, "confirmation"):
        return _ok(Action.SEND_EMAIL, "Confirmation already sent", {"kind": "confirmation", "skipped": True})

    from ..storage import get_email_body
    body_row = get_email_body(email_id) or {}
    to = body_row.get("sender", "")
    subject = subject_override or f"Re: {body_row.get('subject', '')}"

    if _is_mocked():
        sent = {"kind": "confirmation", "to": to, "subject": subject, "success": True,
                "at": _utcnow(), "mock": True}
    else:
        from .. import gmail
        result = gmail.send_email(to, subject, body, thread_id=body_row.get("thread_id"))
        sent = {"kind": "confirmation", "to": to, "subject": subject, "success": True,
                "at": _utcnow(), "gmail_message_id": result.get("id")}

    persistence.append_email_list_field(email_id, "external_actions", sent)
    return _ok(Action.SEND_EMAIL, f"Confirmation sent to {to}", {"kind": "confirmation", "to": to})


def batch_label(email_ids: list[str], category: str) -> list[ToolResult]:
    return [label_email(eid, category) for eid in email_ids]


def batch_archive(email_ids: list[str], folder: str = "archive") -> list[ToolResult]:
    return [archive_email(eid, folder) for eid in email_ids]
