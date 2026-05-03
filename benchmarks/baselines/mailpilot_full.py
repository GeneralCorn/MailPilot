from __future__ import annotations

from triage import persistence
from triage.pipeline import run_pipeline
from triage.schemas import Message


def _derive_action(state_row: dict, in_needs_review: bool) -> str:
    """Reduce the agent's multi-faceted output to a single comparable action label."""
    escalations = state_row.get("escalations") or []
    proposed = state_row.get("proposed_actions") or []
    external = state_row.get("external_actions") or []
    flagged = bool(state_row.get("flagged", 0))
    draft_reply = state_row.get("draft_reply")
    calendar = state_row.get("calendar_event")

    if escalations or in_needs_review:
        return "escalate"

    proposed_tools = [a.get("tool") for a in proposed if isinstance(a, dict)]
    if "calendar" in proposed_tools:
        return "calendar"
    if "send_email" in proposed_tools or "reply_draft" in proposed_tools or draft_reply:
        return "reply_draft"

    executed_tools = [a.get("tool") for a in external if isinstance(a, dict)]
    if "archive" in executed_tools:
        return "archive"
    if "label" in executed_tools:
        return "label"

    if flagged:
        return "flag"
    if calendar:
        return "calendar"
    return "no_action"


def extract_predictions(emails: list[Message], state) -> list[dict]:
    out = []
    for m in emails:
        eid = m.id
        row = persistence.get_email_state(eid) or {}
        in_needs_review = eid in state.needs_review
        out.append({
            "id": eid,
            "subject": m.subject,
            "category": row.get("category") or "unclassified",
            "priority": row.get("priority") or "medium",
            "action": _derive_action(row, in_needs_review),
            "needs_review": in_needs_review,
            "proposed_actions": row.get("proposed_actions") or [],
            "external_actions": row.get("external_actions") or [],
        })
    return out


def run(emails: list[Message], *, evaluator_fn=None) -> list[dict]:
    state, _run_id = run_pipeline(emails, evaluator_fn=evaluator_fn)
    return extract_predictions(emails, state)
