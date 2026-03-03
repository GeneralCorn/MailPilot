from django.db import models


CATEGORY_CHOICES = [
    ("marketing", "Marketing"),
    ("personal", "Personal"),
    ("work", "Work"),
    ("risk", "Risk"),
    ("billing", "Billing"),
    ("unclassified", "Unclassified"),
]

PRIORITY_CHOICES = [
    (1, "Critical"),
    (2, "High"),
    (3, "Medium"),
    (4, "Low"),
    (5, "Minimal"),
]

STATUS_CHOICES = [
    ("pending", "Pending"),
    ("processing", "Processing"),
    ("done", "Done"),
    ("failed", "Failed"),
    ("needs_review", "Needs Review"),
]

ACTION_CHOICES = [
    ("label", "Label"),
    ("flag", "Flag"),
    ("archive", "Archive"),
    ("reply_draft", "Draft Reply"),
    ("calendar", "Add to Calendar"),
    ("escalate", "Escalate"),
    ("no_action", "No Action"),
]


class Email(models.Model):
    """Represents a single email in the triage inbox."""
    subject = models.CharField(max_length=500)
    sender = models.EmailField()
    sender_name = models.CharField(max_length=200, blank=True)
    recipient = models.EmailField(blank=True)
    body = models.TextField()
    snippet = models.CharField(max_length=300, blank=True)
    received_at = models.DateTimeField()
    thread_id = models.CharField(max_length=100, blank=True)
    source = models.CharField(max_length=20, default="manual",
                              choices=[("gmail", "Gmail"), ("outlook", "Outlook"), ("manual", "Manual")])

    # Agent-assigned fields
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default="unclassified")
    priority = models.IntegerField(choices=PRIORITY_CHOICES, default=3)
    confidence = models.FloatField(default=0.0)
    risk_score = models.FloatField(default=0.0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    explanation = models.TextField(blank=True)

    class Meta:
        ordering = ["priority", "-received_at"]

    def __str__(self):
        return f"[{self.get_category_display()}] {self.subject}"


class Task(models.Model):
    """An action the agent proposes for an email."""
    email = models.ForeignKey(Email, on_delete=models.CASCADE, related_name="tasks")
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    parameters = models.JSONField(default=dict, blank=True)
    reason = models.TextField(blank=True)
    approved = models.BooleanField(default=False)
    executed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_action_display()} → {self.email.subject[:40]}"
