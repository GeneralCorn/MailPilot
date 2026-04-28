import json
import os
from datetime import datetime, timezone

import anthropic

from triage.schemas import AgentMessage, Category, Priority, State

from ._anthropic import call_tools
from ._caching import to_cached_request

_MODEL = "claude-sonnet-4-6"
MAX_BATCH = 100

_SYSTEM = (
    "You are the Ranker in MailPilot. Given a batch of classified emails, assign each "
    "one a priority tier and a numeric score in [0, 1] using the rank_emails tool.\n\n"
    "Priority tiers (highest first): urgent, important, normal, low, minimal.\n"
    "Score: 1.0 = top of queue, 0.0 = bottom.\n\n"
    "Guidance:\n"
    "- Urgency cues (deadlines, RSVPs, time-sensitive asks) raise priority.\n"
    "- Sender role and category matter: work > billing > personal > marketing.\n"
    "- Emails classified as 'risk' get low/minimal priority — they will be flagged for\n"
    "  human review, not acted on automatically, so they should not block the queue.\n"
    "- Keep scores spread across the [0, 1] range so the queue is well-ordered."
)


_TOOL = {
    "name": "rank_emails",
    "description": "Submit the ranking of a batch of classified emails.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ranked": {
                "type": "array",
                "description": "One entry per email, in any order — caller will sort by score",
                "items": {
                    "type": "object",
                    "properties": {
                        "email_id": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "priority": {
                            "type": "string",
                            "enum": [p.value for p in Priority],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["email_id", "score", "priority", "reason"],
                },
            }
        },
        "required": ["ranked"],
    },
}


_FEWSHOT_INPUT_1 = json.dumps(
    [
        {"id": "e1", "subject": "Q3 planning — please RSVP by EOD",
         "sender": "manager@acme.com", "category": "work",
         "confidence": 0.9, "risk_score": 0.05},
        {"id": "e2", "subject": "Black Friday — 60% off everything",
         "sender": "deals@shop.com", "category": "marketing",
         "confidence": 0.96, "risk_score": 0.02},
        {"id": "e3", "subject": "Invoice #4421 — paid",
         "sender": "billing@vendor.com", "category": "billing",
         "confidence": 0.93, "risk_score": 0.04},
    ],
    indent=2,
)

_FEWSHOT_OUTPUT_1 = (
    "rank_emails(ranked=[\n"
    "  {email_id: 'e1', score: 0.92, priority: 'urgent', reason: 'RSVP needed by end of day'},\n"
    "  {email_id: 'e3', score: 0.45, priority: 'normal', reason: 'routine paid receipt'},\n"
    "  {email_id: 'e2', score: 0.08, priority: 'minimal', reason: 'promotional, no action expected'}\n"
    "])"
)

_FEWSHOT_INPUT_2 = json.dumps(
    [
        {"id": "a", "subject": "Project status update",
         "sender": "lead@acme.com", "category": "work",
         "confidence": 0.88, "risk_score": 0.04},
        {"id": "b", "subject": "URGENT: Verify your account NOW",
         "sender": "support@secure-bank-verify.ru", "category": "risk",
         "confidence": 0.94, "risk_score": 0.96},
        {"id": "c", "subject": "Coffee Friday?",
         "sender": "alex@gmail.com", "category": "personal",
         "confidence": 0.85, "risk_score": 0.02},
    ],
    indent=2,
)

_FEWSHOT_OUTPUT_2 = (
    "rank_emails(ranked=[\n"
    "  {email_id: 'a', score: 0.7, priority: 'important', reason: 'work status, not deadline-bound'},\n"
    "  {email_id: 'c', score: 0.35, priority: 'low', reason: 'casual social, no time pressure'},\n"
    "  {email_id: 'b', score: 0.1, priority: 'minimal', reason: 'phishing — flagged for review'}\n"
    "])"
)


def _build_messages(batch_json: str) -> list[AgentMessage]:
    return [
        AgentMessage(role="system", content=_SYSTEM),
        AgentMessage(role="user", content="Rank these emails:\n" + _FEWSHOT_INPUT_1),
        AgentMessage(role="assistant", content=_FEWSHOT_OUTPUT_1),
        AgentMessage(role="user", content="Rank these emails:\n" + _FEWSHOT_INPUT_2),
        AgentMessage(role="assistant", content=_FEWSHOT_OUTPUT_2),
        AgentMessage(role="user", content="Rank these emails:\n" + batch_json),
    ]


def _build_batch(state: State, emails) -> str:
    rows = []
    for m in emails:
        rows.append(
            {
                "id": m.id,
                "subject": m.subject,
                "sender": m.sender,
                "category": state.classifications.get(m.id, Category.UNCLASSIFIED).value,
                "confidence": round(state.confidence_scores.get(m.id, 0.0), 2),
                "risk_score": round(state.risk_scores.get(m.id, 0.0), 2),
                "received_at": m.received_at.isoformat() if m.received_at else None,
            }
        )
    return json.dumps(rows, indent=2)


def _fallback(state: State, emails) -> None:
    """Sort by received_at desc, mark everyone NORMAL, push all to needs_review."""
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    ordered = sorted(emails, key=lambda m: m.received_at or epoch, reverse=True)
    state.priority_queue = []
    for m in ordered:
        state.priorities[m.id] = Priority.NORMAL
        state.priority_queue.append((m.id, 0.5))
        if m.id not in state.needs_review:
            state.needs_review.append(m.id)


def rank(state: State) -> None:
    if not state.messages:
        state.priority_queue = []
        return

    # single-email batches don't need an LLM call; nothing to compare against
    if len(state.messages) == 1:
        m = state.messages[0]
        state.priorities[m.id] = Priority.NORMAL
        state.priority_queue = [(m.id, 0.5)]
        return

    batch = state.messages[:MAX_BATCH]
    overflow = state.messages[MAX_BATCH:]
    for m in overflow:
        state.priorities[m.id] = Priority.NORMAL

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_msgs = _build_messages(_build_batch(state, batch))
    system, messages = to_cached_request(all_msgs)

    result = call_tools(
        client,
        model=_MODEL,
        system=system,
        messages=messages,
        tools=[_TOOL],
        force_tool="rank_emails",
        max_tokens=4096,
    )

    if not result:
        _fallback(state, batch + overflow)
        return

    try:
        ranked = result[0]["input"]["ranked"]
        queue: list[tuple[str, float]] = []
        for entry in ranked:
            eid = str(entry["email_id"])
            score = max(0.0, min(1.0, float(entry.get("score", 0.5))))
            priority = Priority(entry.get("priority", "normal"))
            state.priorities[eid] = priority
            queue.append((eid, score))
        queue.sort(key=lambda x: -x[1])
        for m in overflow:
            queue.append((m.id, 0.0))
        state.priority_queue = queue
    except (KeyError, ValueError, TypeError):
        _fallback(state, batch + overflow)
