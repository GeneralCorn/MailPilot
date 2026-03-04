from .storage import _load, _save

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .schemas import Message, Priority

# Lower number = higher priority in sort order
_TIER_ORDER = {p.value: i for i, p in enumerate(Priority)}
_DEFAULT_TIER = len(Priority)

def inbox(request):
    emails = _load()
    category = request.GET.get("category")
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
        "tasks": [],
        "current_category": category or "all",
    })


def email_detail(request, idx):
    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    return JsonResponse({"idx": idx, **emails[idx]})


@require_POST
def triage_email(request, idx):
    # TODO: call agent pipeline
    emails = _load()
    if idx >= len(emails):
        return JsonResponse({"error": "not found"}, status=404)
    return JsonResponse({"status": "ok", "idx": idx})


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
    # TODO: call agent.run_pipeline
    emails = _load()
    return JsonResponse({"status": "pipeline_started", "count": len(emails)})
