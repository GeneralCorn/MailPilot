from .storage import _load, _save, load_inbox

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .schemas import Message, Priority, State
from . import pipeline as pipeline_mod

# Lower number = higher priority in sort order
_TIER_ORDER = {p.value: i for i, p in enumerate(Priority)}
_DEFAULT_TIER = len(Priority)

_ATTENTION_STATUSES = {"flagged", "partial_done", "pending"}


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
        return (tier, rank)

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
    """Emails that worked but need follow-up: flagged / partial_done / pending."""
    out: list[dict] = []
    for i, e in enumerate(emails):
        status = (e.get("status") or "").strip()
        escalations = e.get("escalations") or []
        notes = e.get("notes") or []
        if status in _ATTENTION_STATUSES or escalations:
            label = status or ("flagged" if escalations else "")
            reason = ""
            if escalations:
                reason = escalations[-1].get("reason", "")
            elif notes:
                reason = notes[-1].get("text", "")
            out.append({
                "idx": i,
                "subject": e.get("subject", "(no subject)"),
                "sender": e.get("sender", ""),
                "status_label": label,
                "category": e.get("category", ""),
                "reason": reason,
            })
    return out


def email_detail(request, idx):
    emails = load_inbox()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    return JsonResponse({"idx": idx, **emails[idx]})


@require_POST
def triage_email(request, idx):
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JsonResponse(
            {"error": "ANTHROPIC_API_KEY is not set; pipeline cannot run"}, status=400
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
