from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import anthropic

from triage.schemas import (
    Action,
    Category,
    Message,
    State,
    Status,
    ToolCall,
    ToolResult,
)

from ._anthropic import call_tools

if TYPE_CHECKING:
    from triage.runtime import Runtime

_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
MAX_ITERATIONS = 5

# tools whose successful execution implies the email still needs human attention
_HUMAN_ATTENTION_TOOLS = {Action.ESCALATE, Action.FLAG}

# tools that require explicit user approval before they fire (irreversible side-effects)
_NEEDS_APPROVAL = {Action.CALENDAR, Action.SEND_EMAIL}


def _split_plan(plan: list[ToolCall]) -> tuple[list[ToolCall], list[ToolCall]]:
    auto = [c for c in plan if c.tool not in _NEEDS_APPROVAL]
    proposed = [c for c in plan if c.tool in _NEEDS_APPROVAL]
    return auto, proposed


_SYSTEM = (
    "You are the Worker in MailPilot. Given one classified email, decide what tool calls "
    "to make to handle it. You can call multiple tools in a single response.\n\n"
    "Guidance:\n"
    "- For work and risk emails, ALWAYS start with a `summarize` call.\n"
    "- Risk emails must be escalated, never auto-replied or auto-archived.\n"
    "- Marketing emails: `archive` into 'promotions' is usually enough.\n"
    "- When a meeting email contains an RSVP signal (phrases like 'please confirm', 'RSVP',\n"
    "  'let me know if you can attend'), ALWAYS call BOTH `calendar` and `send_email` (kind='rsvp').\n"
    "  These two tools are gated behind a human approval UI — the user decides whether to\n"
    "  actually send / book, so do not pre-filter on your own. Skipping send_email here is wrong.\n"
    "- Personal: prefer `reply_draft` over auto-send unless an explicit RSVP is requested.\n"
    "- Be conservative ONLY for irreversible actions outside the proposal queue (label, archive,\n"
    "  escalate). For calendar / send_email always include them when the email content suggests\n"
    "  them — the user can reject in the UI.\n"
    "- The runtime fills in `email_id`; don't supply it yourself."
)


_TOOLS: list[dict] = [
    {
        "name": "summarize",
        "description": "Save a 1-2 sentence summary of the email body for the reviewer.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
    {
        "name": "label",
        "description": "Re-label the email's category. Rare; the router has done this.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": [c.value for c in Category]}},
            "required": ["category"],
        },
    },
    {
        "name": "flag",
        "description": "Mark the email as needing user attention.",
        "input_schema": {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        },
    },
    {
        "name": "archive",
        "description": "Move the email into a folder when no follow-up is needed.",
        "input_schema": {
            "type": "object",
            "properties": {"folder": {"type": "string"}},
            "required": ["folder"],
        },
    },
    {
        "name": "reply_draft",
        "description": "Save a draft reply to Gmail for human review (NOT sent).",
        "input_schema": {
            "type": "object",
            "properties": {
                "body": {"type": "string"},
                "subject_override": {"type": "string"},
            },
            "required": ["body"],
        },
    },
    {
        "name": "calendar",
        "description": (
            "Propose a Google Calendar event. The user will review and approve before "
            "the event is actually created."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO 8601"},
                "end_time": {"type": "string", "description": "ISO 8601"},
                "location": {"type": "string"},
                "response": {"type": "string", "enum": ["accept", "decline", "tentative"]},
            },
            "required": ["title", "start_time", "end_time"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Propose sending a Gmail message (RSVP or confirmation). The user will review "
            "and approve before any email is actually sent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["rsvp", "confirmation"]},
                "body": {"type": "string"},
                "decision": {"type": "string", "enum": ["accept", "decline", "tentative"]},
                "subject_override": {"type": "string"},
            },
            "required": ["kind", "body"],
        },
    },
    {
        "name": "escalate",
        "description": "Route a risky or ambiguous email to a human reviewer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["target", "reason"],
        },
    },
    {
        "name": "no_action",
        "description": "Explicitly record that no action is needed.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


def _fewshot_messages() -> list[dict]:
    """Three few-shots in proper tool_use + tool_result form, with cache_control on the last block."""
    return [
        # marketing → archive
        {"role": "user", "content": (
            "Subject: Big sale — 50% off this weekend\n"
            "Sender: deals@store.com\n"
            "Category: marketing\n"
            "Body: Limited-time offers on everything in store."
        )},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs1_a", "name": "archive",
             "input": {"folder": "promotions"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs1_a", "content": "Moved to 'promotions'"},
        ]},

        # work meeting → summarize + calendar + send_email(rsvp)
        {"role": "user", "content": (
            "Subject: Standup Friday 10:00 AM\n"
            "Sender: lead@acme.com\n"
            "Category: work\n"
            "Body: Quick standup Friday 10–10:30 in Room 4. Please confirm."
        )},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs2_a", "name": "summarize",
             "input": {"summary": "Standup Friday 10:00–10:30 AM in Room 4."}},
            {"type": "tool_use", "id": "fs2_b", "name": "calendar",
             "input": {
                 "title": "Standup",
                 "start_time": "2026-04-30T10:00:00Z",
                 "end_time": "2026-04-30T10:30:00Z",
                 "location": "Room 4",
                 "response": "accept",
             }},
            {"type": "tool_use", "id": "fs2_c", "name": "send_email",
             "input": {
                 "kind": "rsvp",
                 "decision": "accept",
                 "body": "See you Friday.",
             }},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs2_a", "content": "Summary saved"},
            {"type": "tool_result", "tool_use_id": "fs2_b", "content": "Event proposal queued for approval"},
            {"type": "tool_result", "tool_use_id": "fs2_c", "content": "RSVP proposal queued for approval"},
        ]},

        # risk → summarize + escalate
        {"role": "user", "content": (
            "Subject: URGENT: Verify your account NOW\n"
            "Sender: support@bank-verify.ru\n"
            "Category: risk\n"
            "Body: Click this link immediately to keep your account active."
        )},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fs3_a", "name": "summarize",
             "input": {"summary": "Likely phishing — fake verification request from suspicious .ru domain."}},
            {"type": "tool_use", "id": "fs3_b", "name": "escalate",
             "input": {"target": "security", "reason": "phishing indicators"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "fs3_a", "content": "Summary saved"},
            # cache_control on the LAST few-shot block — caches system + tools + all few-shots
            {"type": "tool_result", "tool_use_id": "fs3_b", "content": "Escalated to security",
             "cache_control": {"type": "ephemeral"}},
        ]},
    ]


def _target_user_message(email: Message, category: Category, prior_results: list[ToolResult] | None) -> dict:
    body = email.body_plain or email.snippet or ""
    text = (
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Category: {category.value}\n"
        f"Body: {body[:600]}"
    )
    if prior_results:
        prior_lines = "\n".join(
            f"- {r.tool.value}: {'OK' if r.success else 'FAILED — ' + r.message[:120]}"
            for r in prior_results
        )
        text += (
            f"\n\nActions already attempted in this run:\n{prior_lines}\n\n"
            "Replan the remaining steps to recover. Skip any actions already done."
        )
    return {"role": "user", "content": text}


def plan_actions(
    email: Message, state: State, prior_results: list[ToolResult] | None = None
) -> list[ToolCall]:
    category = state.classifications.get(email.id, Category.UNCLASSIFIED)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = _fewshot_messages() + [_target_user_message(email, category, prior_results)]

    result = call_tools(
        client,
        model=_MODEL,
        system=_SYSTEM,
        messages=messages,
        tools=_TOOLS,
        max_tokens=2048,
    )

    if result is None:
        return [
            ToolCall(
                tool=Action.ESCALATE,
                parameters={"email_id": email.id, "target": "human", "reason": "worker planner failed"},
                reason="planner LLM failure",
            )
        ]

    calls: list[ToolCall] = []
    for entry in result:
        try:
            tool = Action(entry["name"])
        except ValueError:
            continue
        params = dict(entry.get("input") or {})
        params["email_id"] = email.id
        calls.append(ToolCall(tool=tool, parameters=params, reason=""))

    if not calls:
        return [
            ToolCall(
                tool=Action.ESCALATE,
                parameters={"email_id": email.id, "target": "human", "reason": "worker produced no valid actions"},
                reason="planner returned no usable tools",
            )
        ]
    return calls


def classify_error(result: ToolResult) -> str:
    """retryable | recoverable | confirmation | fatal — bucketed from registry-wrapped ToolResult."""
    err = (result.data.get("error_type") or "").strip()
    msg = (result.message or "").lower()

    if "permission" in msg or "unauthorized" in msg or "scope" in msg or "insufficient" in msg:
        return "confirmation"
    if err == "PermissionError":
        return "confirmation"

    is_http = "http" in err.lower() and "error" in err.lower()
    if is_http or "httperror" in msg:
        if "401" in msg or "403" in msg:
            return "confirmation"
        if any(c in msg for c in (" 500", " 502", " 503", " 504")):
            return "retryable"
        return "recoverable"

    if err in {"TimeoutError", "ConnectionError", "ServerNotFoundError"}:
        return "retryable"
    if "timeout" in msg or "connection refused" in msg:
        return "retryable"

    if "quota" in msg or "rate limit" in msg:
        return "recoverable"

    return "fatal"


def _backoff(retries: int) -> None:
    time.sleep(min(0.5 * (2 ** (retries - 1)), 4.0))


def work(email: Message, state: State, runtime: "Runtime | None" = None) -> Status:
    if runtime is None:
        from triage.runtime import Runtime as _Runtime
        runtime = _Runtime()

    initial_plan: list[ToolCall] = list(plan_actions(email, state))
    state.worker_actions[email.id] = list(initial_plan)

    auto, proposed = _split_plan(initial_plan)
    remaining: list[ToolCall] = list(auto)
    proposed_pile: list[ToolCall] = list(proposed)

    results: list[ToolResult] = []
    succeeded = 0
    total = len(remaining)
    iterations = 0

    while remaining and iterations < MAX_ITERATIONS:
        call = remaining.pop(0)
        retries = 0
        outcome: str | None = None

        while True:
            r = runtime.run_tool(call)
            if r.success:
                results.append(r)
                succeeded += 1
                outcome = "ok"
                break

            results.append(r)
            kind = classify_error(r)

            if kind == "retryable":
                retries += 1
                if retries > MAX_RETRIES:
                    outcome = "exhausted"
                    break
                _backoff(retries)
                continue
            if kind == "recoverable":
                outcome = "replan"
                break
            if kind == "confirmation":
                state.sub_action_results[email.id] = results
                state.email_status[email.id] = Status.FLAGGED
                return Status.FLAGGED
            # fatal
            state.sub_action_results[email.id] = results
            state.email_status[email.id] = Status.PENDING
            return Status.PENDING

        if outcome == "exhausted":
            # let final scoring decide: prior successes -> PARTIAL_DONE, none -> PENDING
            break
        if outcome == "replan":
            iterations += 1
            replanned = list(plan_actions(email, state, prior_results=list(results)))
            new_auto, new_proposed = _split_plan(replanned)
            remaining = new_auto
            proposed_pile.extend(new_proposed)
            total = succeeded + len(remaining)
            continue

    state.sub_action_results[email.id] = results

    if total == 0:
        final = Status.DONE
    elif succeeded == total:
        final = Status.DONE
    elif succeeded > 0:
        final = Status.PARTIAL_DONE
    else:
        final = Status.PENDING

    # escalate/flag tools imply human attention even if all sub-actions succeeded
    if final == Status.DONE and any(
        r.success and r.tool in _HUMAN_ATTENTION_TOOLS for r in results
    ):
        final = Status.FLAGGED

    if proposed_pile:
        state.proposed_actions[email.id] = list(proposed_pile)
        # awaiting_approval takes precedence — there are still actions for the user to review
        final = Status.AWAITING_APPROVAL

    state.email_status[email.id] = final
    return final
