from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import anthropic

from triage.schemas import (
    Action,
    AgentMessage,
    Category,
    Message,
    State,
    Status,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from triage.runtime import Runtime

_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
MAX_ITERATIONS = 5

_SYSTEM = (
    "You are the Worker in MailPilot. Given one classified email, plan a list of tool calls "
    "to handle it. Each call is one sub-action.\n\n"
    "Available tools (parameters in JSON):\n"
    "- summarize:  { summary: string }                            — store a 1-2 sentence summary; required first for work and risk\n"
    "- label:      { category: string }                           — re-label (rare; router did this)\n"
    "- flag:       { flag: bool }                                 — mark as needing attention\n"
    "- archive:    { folder: string }                             — move to a folder; use for done items\n"
    "- reply_draft:{ body: string, subject_override?: string }    — save a draft for human review\n"
    "- calendar:   { title, start_time, end_time, location?, response? } — create a calendar event\n"
    "- send_email: { kind: 'rsvp'|'confirmation', body: string, decision?: 'accept'|'decline'|'tentative', subject_override?: string }\n"
    "- escalate:   { target: string, reason: string }             — route to a human reviewer\n"
    "- no_action:  { reason: string }                             — explicitly nothing to do\n\n"
    "Guidance:\n"
    "- For work and risk emails, ALWAYS start with a summarize action.\n"
    "- Risk emails must be escalated, never auto-replied or auto-archived.\n"
    "- Marketing emails: archive into 'promotions' is usually enough.\n"
    "- Personal: prefer reply_draft over auto-send unless an explicit RSVP is requested.\n"
    "- Be conservative: include only actions you are confident the user wants.\n"
    "- Do NOT include the email_id parameter; the runtime fills it in.\n\n"
    "Respond with valid JSON only, no other text:\n"
    '{"actions":[{"tool":"<tool_name>","parameters":{...},"reason":"<brief>"}]}'
)


def _build_messages(
    email: Message, category: Category, prior_results: list[ToolResult] | None
) -> list[AgentMessage]:
    msgs: list[AgentMessage] = [AgentMessage(role="system", content=_SYSTEM)]

    msgs.append(AgentMessage(
        role="user",
        content=(
            "Subject: Big sale — 50% off this weekend\n"
            "Sender: deals@store.com\n"
            "Category: marketing\n"
            "Body: Limited-time offers on everything in store."
        ),
    ))
    msgs.append(AgentMessage(
        role="assistant",
        content='{"actions":[{"tool":"archive","parameters":{"folder":"promotions"},"reason":"promotional, no follow-up"}]}',
    ))

    msgs.append(AgentMessage(
        role="user",
        content=(
            "Subject: Standup Friday 10:00 AM\n"
            "Sender: lead@acme.com\n"
            "Category: work\n"
            "Body: Quick standup Friday 10–10:30 in Room 4. Please confirm."
        ),
    ))
    msgs.append(AgentMessage(
        role="assistant",
        content=(
            '{"actions":['
            '{"tool":"summarize","parameters":{"summary":"Standup Friday 10:00–10:30 AM in Room 4."},"reason":"context"},'
            '{"tool":"calendar","parameters":{"title":"Standup","start_time":"2026-04-30T10:00:00Z",'
            '"end_time":"2026-04-30T10:30:00Z","location":"Room 4","response":"accept"},"reason":"book the slot"},'
            '{"tool":"send_email","parameters":{"kind":"rsvp","decision":"accept","body":"See you Friday."},"reason":"confirm attendance"}'
            ']}'
        ),
    ))

    msgs.append(AgentMessage(
        role="user",
        content=(
            "Subject: URGENT: Verify your account NOW\n"
            "Sender: support@bank-verify.ru\n"
            "Category: risk\n"
            "Body: Click this link immediately to keep your account active."
        ),
    ))
    msgs.append(AgentMessage(
        role="assistant",
        content=(
            '{"actions":['
            '{"tool":"summarize","parameters":{"summary":"Likely phishing — fake verification request from suspicious .ru domain."},"reason":"context"},'
            '{"tool":"escalate","parameters":{"target":"security","reason":"phishing indicators"},"reason":"never auto-act on risk"}'
            ']}'
        ),
    ))

    body = email.body_plain or email.snippet or ""
    target = (
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
        target += (
            f"\n\nActions already attempted in this run:\n{prior_lines}\n\n"
            "Replan the remaining steps to recover. Skip any actions already done."
        )
    msgs.append(AgentMessage(role="user", content=target))
    return msgs


def _parse_actions(raw: str, email_id: str) -> list[ToolCall]:
    data = json.loads(raw)
    out: list[ToolCall] = []
    for entry in data["actions"]:
        tool = Action(entry["tool"])
        params = dict(entry.get("parameters") or {})
        params["email_id"] = email_id
        out.append(ToolCall(tool=tool, parameters=params, reason=entry.get("reason", "")))
    return out


def plan_actions(
    email: Message, state: State, prior_results: list[ToolResult] | None = None
) -> list[ToolCall]:
    category = state.classifications.get(email.id, Category.UNCLASSIFIED)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_msgs = _build_messages(email, category, prior_results)
    system = next(m.content for m in all_msgs if m.role == "system")
    messages = [{"role": m.role, "content": m.content} for m in all_msgs if m.role != "system"]

    raw = ""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=_MODEL, max_tokens=2048, system=system, messages=messages
            )
            raw = response.content[0].text.strip()
            return _parse_actions(raw, email.id)
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON or unknown tool. Respond only with valid JSON:\n"
                            '{"actions":[{"tool":"<name>","parameters":{...},"reason":"..."}]}'
                        ),
                    },
                ]
        except anthropic.APIError:
            break

    return [
        ToolCall(
            tool=Action.ESCALATE,
            parameters={"email_id": email.id, "target": "human", "reason": "worker planner failed"},
            reason="planner LLM failure",
        )
    ]


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

    remaining: list[ToolCall] = list(plan_actions(email, state))
    state.worker_actions[email.id] = list(remaining)

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
            remaining = list(plan_actions(email, state, prior_results=list(results)))
            total = succeeded + len(remaining)
            continue

    state.sub_action_results[email.id] = results

    if total == 0:
        state.email_status[email.id] = Status.DONE
        return Status.DONE
    if succeeded == total:
        state.email_status[email.id] = Status.DONE
        return Status.DONE
    if succeeded > 0:
        state.email_status[email.id] = Status.PARTIAL_DONE
        return Status.PARTIAL_DONE
    state.email_status[email.id] = Status.PENDING
    return Status.PENDING
