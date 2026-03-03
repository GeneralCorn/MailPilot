from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .models import Email, Task


def inbox(request):
    """Main inbox view — left panel email list + right task panel."""
    emails = Email.objects.all()
    category = request.GET.get("category")
    if category:
        emails = emails.filter(category=category)
    selected_id = request.GET.get("selected")
    selected_email = None
    if selected_id:
        selected_email = Email.objects.filter(pk=selected_id).first()
    elif emails.exists():
        selected_email = emails.first()
    tasks = Task.objects.filter(executed=False).select_related("email")[:20]
    return render(request, "triage/inbox.html", {
        "emails": emails,
        "selected": selected_email,
        "tasks": tasks,
        "current_category": category or "all",
    })


def email_detail(request, pk):
    """AJAX endpoint — returns email detail as JSON."""
    email = get_object_or_404(Email, pk=pk)
    return JsonResponse({
        "id": email.id,
        "subject": email.subject,
        "sender": email.sender,
        "sender_name": email.sender_name,
        "body": email.body,
        "received_at": email.received_at.isoformat(),
        "category": email.category,
        "priority": email.priority,
        "confidence": email.confidence,
        "risk_score": email.risk_score,
        "status": email.status,
        "explanation": email.explanation,
        "tasks": list(email.tasks.values("id", "action", "reason", "approved", "executed")),
    })


@require_POST
def triage_email(request, pk):
    """Run the agent pipeline on a single email."""
    # TODO: call agent.route → agent.evaluate → create Tasks
    email = get_object_or_404(Email, pk=pk)
    return JsonResponse({"status": "ok", "email_id": email.id})


@require_POST
def approve_task(request, pk):
    """Approve a proposed task."""
    task = get_object_or_404(Task, pk=pk)
    task.approved = True
    task.save()
    return JsonResponse({"status": "approved", "task_id": task.id})


@require_POST
def execute_task(request, pk):
    """Execute an approved task."""
    task = get_object_or_404(Task, pk=pk)
    if not task.approved:
        return JsonResponse({"error": "Task not approved"}, status=400)
    # TODO: call agent.work
    task.executed = True
    task.save()
    return JsonResponse({"status": "executed", "task_id": task.id})


@require_POST
def import_emails(request):
    """Import emails from Gmail or Outlook."""
    source = request.POST.get("source", "gmail")
    # TODO: OAuth flow + Gmail API / Microsoft Graph API
    return JsonResponse({"status": "import_started", "source": source})


@require_POST
def run_pipeline(request):
    """Run full triage pipeline on all pending emails."""
    # TODO: call agent.run_pipeline on pending emails
    pending = Email.objects.filter(status="pending")
    return JsonResponse({"status": "pipeline_started", "count": pending.count()})
