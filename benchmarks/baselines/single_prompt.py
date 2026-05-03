from __future__ import annotations

import anthropic

from triage.agent._anthropic import call_json
from triage.agent._client import client_kwargs, default_model
from triage.schemas import Message

from benchmarks.scoring.metrics import ACTIONS, CATEGORIES, PRIORITIES

_SYSTEM = (
    "You are an email triage assistant. Given one email, decide its category, "
    "priority tier, and recommended action. Submit via the submit tool.\n\n"
    f"Categories: {', '.join(CATEGORIES)}\n"
    f"Priority tiers: {', '.join(PRIORITIES)}\n"
    f"Actions: {', '.join(ACTIONS)}\n"
    "needs_review must be true for risk emails (and false otherwise)."
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category":     {"type": "string", "enum": list(CATEGORIES)},
        "priority":     {"type": "string", "enum": list(PRIORITIES)},
        "action":       {"type": "string", "enum": list(ACTIONS)},
        "needs_review": {"type": "boolean"},
    },
    "required": ["category", "priority", "action", "needs_review"],
}

_FALLBACK = {
    "category": "marketing",
    "priority": "low",
    "action": "no_action",
    "needs_review": False,
}


def run(emails: list[Message]) -> list[dict]:
    client = anthropic.Anthropic(**client_kwargs())
    out: list[dict] = []
    for m in emails:
        user = (
            f"Subject: {m.subject}\n"
            f"Sender: {m.sender}\n"
            f"Body: {m.body_plain or m.snippet}"
        )
        result = call_json(
            client,
            model=default_model(),
            system=_SYSTEM,
            user=user,
            output_schema=_OUTPUT_SCHEMA,
        )
        row = result if result is not None else dict(_FALLBACK)
        out.append({
            "id": m.id,
            "category": row.get("category", _FALLBACK["category"]),
            "priority": row.get("priority", _FALLBACK["priority"]),
            "action": row.get("action", _FALLBACK["action"]),
            "needs_review": bool(row.get("needs_review", False)),
            "proposed_actions": [],
            "external_actions": [],
        })
    return out
