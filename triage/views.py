from .storage import _load, _save, _load_traces, _save_trace

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .schemas import Message

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

    if not category:
        email_list = [{"idx": i, **e} for i, e in enumerate(emails)]
    else:
        email_list = [{"idx": emails.index(e), **e} for e in filtered]
    email_list.sort(key=lambda e: e.get("received_at", ""), reverse=True)

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
def delete_emails(request):
    import json as _json
    data = _json.loads(request.body)
    indices = sorted(data.get("indices", []), reverse=True)
    emails = _load()
    for idx in indices:
        if 0 <= idx < len(emails):
            emails.pop(idx)
    _save(emails)
    return JsonResponse({"status": "ok", "remaining": len(emails)})


def _triage_and_save(new_emails: list[dict], all_emails: list[dict]) -> None:
    """Run triage on new emails, saving after each so UI updates incrementally."""
    import time
    from datetime import datetime, timezone
    from .runtime.runtime import Runtime
    from .runtime.agents import register_triage_agents
    from .runtime.message import DispatchMessage

    if not new_emails:
        return

    print(f"\n[triage] Starting triage for {len(new_emails)} emails...")
    rt = Runtime()
    register_triage_agents(rt)

    results = {}
    for i, raw in enumerate(new_emails):
        subj = raw.get("subject", "")[:60]
        eid = raw.get("id", "")

        t0 = time.time()
        print(f"[triage] [{i+1}/{len(new_emails)}] Routing: {subj}")
        rt.dispatch(DispatchMessage(target="router", payload={"email": raw}))
        ps = rt.state.get_artifact("pipeline_state")
        cat = ps.classifications.get(eid)
        print(f"[triage]   -> {cat.value if cat else '?'} ({time.time()-t0:.1f}s)")

        t1 = time.time()
        print(f"[triage] [{i+1}/{len(new_emails)}] Evaluating: {subj}")
        rt.dispatch(DispatchMessage(target="evaluator", payload={"email": raw}))
        ps = rt.state.get_artifact("pipeline_state")
        cat = ps.classifications.get(eid)
        risk = ps.risk_scores.get(eid, 0)
        review = eid in ps.needs_review
        print(f"[triage]   -> {cat.value if cat else '?'} (risk={risk:.2f}, review={review}, {time.time()-t1:.1f}s)")

        # Update email dict and save immediately so UI reflects it
        raw["category"] = cat.value if cat else "unclassified"
        if review:
            raw["needs_review"] = True
        else:
            raw.pop("needs_review", None)
        _save(all_emails)

        results[eid] = {
            "subject": raw.get("subject", ""),
            "category": raw["category"],
            "confidence": ps.confidence_scores.get(eid, 0),
            "risk_score": risk,
            "needs_review": review,
        }
        print(f"[triage] [{i+1}/{len(new_emails)}] Saved. Total: {time.time()-t0:.1f}s")

    _save_trace({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "email_count": len(new_emails),
        "results": results,
        "events": rt.trace_to_list(),
    })
    print(f"[triage] Done. {len(new_emails)} emails triaged.\n")


def traces(request):
    trace_list = _load_traces()
    trace_list.reverse()  # newest first
    return render(request, "triage/traces.html", {"traces": trace_list})


@require_POST
def import_emails(request):
    from .gmail import fetch_emails
    try:
        raw = fetch_emails(max_results=20)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    emails = _load()
    existing_threads = {e.get("thread_id") for e in emails}
    new_emails = []
    for data in raw:
        if data.get("thread_id") not in existing_threads:
            msg = Message(**data)
            email_dict = msg.model_dump(mode="json")
            emails.append(email_dict)
            new_emails.append(email_dict)
            existing_threads.add(data.get("thread_id"))

    _save(emails)

    # Triage in background thread so the response returns immediately
    if new_emails:
        import threading
        threading.Thread(target=_triage_and_save, args=(new_emails, emails), daemon=True).start()

    return JsonResponse({"status": "imported", "new": len(new_emails), "total": len(raw)})
