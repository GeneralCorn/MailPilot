from datetime import datetime

from .storage import _load, _save, load_inbox

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .schemas import Message, Priority, State
from . import pipeline as pipeline_mod

# Lower number = higher priority in sort order
_TIER_ORDER = {p.value: i for i, p in enumerate(Priority)}
_DEFAULT_TIER = len(Priority)

_ATTENTION_STATUSES = {"awaiting_approval", "flagged", "partial_done", "pending"}
_PROCESSED_STATUSES = _ATTENTION_STATUSES | {"done"}
_TASK_QUEUE_LIMIT = 20

# higher priority -> earlier in task queue
_TASK_PRIORITY = {
    "awaiting_approval": 0,
    "flagged": 1,
    "partial_done": 2,
    "pending": 3,
    "done": 4,
}


def _to_timestamp(value) -> float:
    """ISO string or datetime -> unix seconds; missing/bad value -> 0 (sorts last when negated)."""
    if not value:
        return 0.0
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return 0.0


def inbox(request):
    emails = load_inbox()
    category = request.GET.get("category")
    tasks = _build_task_queue(emails)
    if category:
        filtered = [e for e in emails if e.get("category", "unclassified") == category]
    else:
        filtered = emails

    selected_idx = request.GET.get("selected")
    selected = None
    if selected_idx and selected_idx.isdigit() and int(selected_idx) < len(emails):
        selected = {**emails[int(selected_idx)], "idx": int(selected_idx)}
    elif filtered:
        real_idx = emails.index(filtered[0])
        selected = {**filtered[0], "idx": real_idx}

    def _sort_key(item):
        e = item[1] if isinstance(item, tuple) else item
        tier = _TIER_ORDER.get(e.get("priority", ""), _DEFAULT_TIER)
        rank = e.get("priority_rank", 9999)
        return (tier, rank, -_to_timestamp(e.get("received_at")))

    if not category:
        email_list = [{"idx": i, **e} for i, e in enumerate(emails)]
    else:
        email_list = [{"idx": emails.index(e), **e} for e in filtered]
    email_list.sort(key=_sort_key)

    return render(request, "triage/inbox.html", {
        "emails": email_list,
        "selected": selected,
        "tasks": tasks,
        "current_category": category or "all",
    })


def _build_task_queue(emails: list[dict]) -> list[dict]:
    """Activity feed of triaged emails. Order: awaiting_approval > flagged > partial_done > pending > done."""
    items: list[tuple[int, float, dict]] = []
    for i, e in enumerate(emails):
        status = (e.get("status") or "").strip()
        if status not in _PROCESSED_STATUSES:
            continue

        escalations = e.get("escalations") or []
        notes = e.get("notes") or []
        summary = (e.get("summary") or "").strip()
        proposed = e.get("proposed_actions") or []

        if proposed:
            reason = f"{len(proposed)} action(s) awaiting approval"
        elif escalations:
            reason = escalations[-1].get("reason", "")
        elif summary:
            reason = summary
        elif notes:
            reason = notes[-1].get("text", "")
        else:
            reason = ""

        item = {
            "idx": i,
            "subject": e.get("subject", "(no subject)"),
            "sender": e.get("sender", ""),
            "status_label": status,
            "category": e.get("category", ""),
            "reason": reason,
        }
        priority = _TASK_PRIORITY.get(status, 99)
        items.append((priority, -_to_timestamp(e.get("received_at")), item))

    items.sort(key=lambda t: (t[0], t[1]))
    return [it for _, _, it in items[:_TASK_QUEUE_LIMIT]]


def email_detail(request, idx):
    emails = load_inbox()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    return JsonResponse({"idx": idx, **emails[idx]})


@require_POST
def triage_email(request, idx):
    import os
    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        return JsonResponse(
            {"error": "DEEPSEEK_API_KEY is not set (or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic); pipeline cannot run"},
            status=400,
        )
    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)

    msg = Message(**emails[idx])
    try:
        state, run_id = pipeline_mod.run_pipeline([msg])
    except Exception as exc:
        return JsonResponse(
            {"error": f"{type(exc).__name__}: {exc}"}, status=500
        )

    from . import persistence
    state_row = persistence.get_email_state(msg.id)
    final_status = state.email_status.get(msg.id)

    return JsonResponse(
        {
            "run_id": run_id,
            "idx": idx,
            "email_id": msg.id,
            "category": state_row.get("category"),
            "priority": state_row.get("priority"),
            "status": final_status.value if final_status else state_row.get("status"),
            "needs_review": msg.id in state.needs_review,
        }
    )


_RECOMPUTE_FIELDS = ("status",)


def _recompute_email_status(email_id: str):
    """After approve/reject, recompute the final status based on remaining proposals + sub-action history."""
    from . import persistence
    from .agent.worker import _HUMAN_ATTENTION_TOOLS  # reuse existing flag rule
    from .schemas import Action, Status

    row = persistence.get_email_state(email_id)
    proposed = row.get("proposed_actions") or []
    if proposed:
        new_status = Status.AWAITING_APPROVAL.value
        persistence.update_email_state(email_id, status=new_status)
        return new_status

    # all proposals resolved — derive from external_actions (the run-history of approved tools)
    history = row.get("external_actions") or []
    successes = [a for a in history if a.get("success")]
    fired_human = any(
        a.get("tool") in {Action.ESCALATE.value, Action.FLAG.value} for a in history
    )
    if not history:
        new_status = Status.DONE.value
    elif len(successes) == len(history):
        new_status = Status.FLAGGED.value if fired_human else Status.DONE.value
    elif successes:
        new_status = Status.PARTIAL_DONE.value
    else:
        new_status = Status.PENDING.value
    persistence.update_email_state(email_id, status=new_status)
    return new_status


def _record_external_action(email_id: str, tool: str, success: bool, summary: str) -> None:
    """Track an approved/executed proposal in email_state.external_actions for status recomputation."""
    from . import persistence
    persistence.append_email_list_field(email_id, "external_actions", {
        "tool": tool,
        "success": success,
        "summary": summary,
        "at": datetime.now().isoformat(),
        "via": "approval",
    })


def proposals_list(request, idx):
    """GET — return the current pending proposals for an email."""
    from . import persistence
    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    email_id = emails[idx]["id"]
    row = persistence.get_email_state(email_id)
    return JsonResponse({
        "email_id": email_id,
        "status": row.get("status"),
        "proposed_actions": row.get("proposed_actions") or [],
    })


@require_POST
def proposal_approve(request, idx, proposal_idx):
    """POST — execute the i-th proposal after merging any user-edited query-string params."""
    from . import persistence
    from .runtime import Runtime
    from .schemas import ToolCall

    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    email_id = emails[idx]["id"]

    # collect user-edited overrides from query string (skip empty values)
    overrides = {k: v for k, v in request.GET.items() if v != ""}
    if overrides:
        updated = persistence.update_proposed_action(email_id, proposal_idx, overrides)
        if updated is None:
            return JsonResponse({"error": "proposal index out of range"}, status=404)

    removed = persistence.remove_proposed_action(email_id, proposal_idx)
    if removed is None:
        return JsonResponse({"error": "proposal index out of range"}, status=404)

    params = dict(removed.get("parameters") or {})
    params.setdefault("email_id", email_id)
    try:
        call = ToolCall(tool=removed["tool"], parameters=params, reason=removed.get("reason", ""))
    except Exception as exc:
        return JsonResponse({"error": f"bad ToolCall: {exc}"}, status=400)

    try:
        result = Runtime().run_tool(call)
    except Exception as exc:
        _record_external_action(email_id, removed["tool"], False, f"{type(exc).__name__}: {exc}")
        _recompute_email_status(email_id)
        return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    _record_external_action(email_id, removed["tool"], result.success, result.message)
    new_status = _recompute_email_status(email_id)

    return JsonResponse({
        "ok": result.success,
        "tool": call.tool.value,
        "message": result.message,
        "data": result.data,
        "new_status": new_status,
    })


@require_POST
def proposal_reject(request, idx, proposal_idx):
    """POST — discard the i-th proposal without executing."""
    from . import persistence
    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    email_id = emails[idx]["id"]
    removed = persistence.remove_proposed_action(email_id, proposal_idx)
    if removed is None:
        return JsonResponse({"error": "proposal index out of range"}, status=404)
    new_status = _recompute_email_status(email_id)
    return JsonResponse({
        "ok": True,
        "rejected_tool": removed.get("tool"),
        "new_status": new_status,
    })


@require_POST
def import_emails(request):
    from .gmail import fetch_emails
    try:
        raw = fetch_emails(max_results=20)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    emails = _load()
    existing_threads = {e.get("thread_id") for e in emails}
    count = 0
    for data in raw:
        if data.get("thread_id") not in existing_threads:
            msg = Message(**data)
            emails.append(msg.model_dump(mode="json"))
            existing_threads.add(data.get("thread_id"))
            count += 1
    _save(emails)
    return JsonResponse({"status": "imported", "new": count, "total": len(raw)})


@require_POST
def run_pipeline(request):
    raw = _load()
    if not raw:
        return JsonResponse({"error": "no emails to process"}, status=400)
    messages = [Message(**e) for e in raw]
    state, run_id = pipeline_mod.run_pipeline(messages)
    by_status: dict[str, int] = {}
    for status in state.email_status.values():
        key = status.value if hasattr(status, "value") else str(status)
        by_status[key] = by_status.get(key, 0) + 1
    return JsonResponse(
        {
            "run_id": run_id,
            "processed_count": len(state.email_status),
            "by_status": by_status,
            "needs_review": list(state.needs_review),
        }
    )
