"""Seed the database with demo emails and tasks for development."""
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from triage.models import Email, Task


DEMO_EMAILS = [
    {
        "subject": "Q1 Revenue Report — Action Required",
        "sender": "cfo@acmecorp.com",
        "sender_name": "Sarah Chen",
        "body": "Hi team,\n\nPlease review the attached Q1 revenue report before our board meeting on Friday. We need sign-off from all department heads by EOD Thursday.\n\nKey highlights:\n- Revenue up 12% YoY\n- Customer churn down to 3.2%\n- APAC expansion ahead of schedule\n\nLet me know if you have questions.\n\nBest,\nSarah",
        "snippet": "Please review the attached Q1 revenue report before our board meeting...",
        "category": "work",
        "priority": 1,
        "confidence": 94.0,
        "status": "pending",
    },
    {
        "subject": "URGENT: Suspicious Login Attempt Detected",
        "sender": "security@company.com",
        "sender_name": "IT Security",
        "body": "We detected a login attempt to your account from an unrecognized device in Lagos, Nigeria at 3:42 AM EST.\n\nIf this was not you, please reset your password immediately and contact IT security.\n\nDevice: Chrome on Windows 10\nIP: 102.89.xx.xx\nLocation: Lagos, Nigeria",
        "snippet": "We detected a login attempt to your account from an unrecognized device...",
        "category": "risk",
        "priority": 1,
        "confidence": 97.0,
        "risk_score": 85.0,
        "status": "needs_review",
    },
    {
        "subject": "Invoice #4821 — Payment Due March 15",
        "sender": "billing@cloudhost.io",
        "sender_name": "CloudHost Billing",
        "body": "Your monthly hosting invoice is ready.\n\nInvoice #4821\nAmount: $2,340.00\nDue Date: March 15, 2026\n\nPlease remit payment to avoid service interruption.\n\nView invoice: https://cloudhost.io/invoices/4821",
        "snippet": "Your monthly hosting invoice is ready. Amount: $2,340.00 Due: March 15",
        "category": "billing",
        "priority": 2,
        "confidence": 91.0,
        "status": "pending",
    },
    {
        "subject": "Team Lunch Friday?",
        "sender": "mike.johnson@gmail.com",
        "sender_name": "Mike Johnson",
        "body": "Hey!\n\nWant to grab lunch Friday at that new ramen place? I heard it's amazing. Let me know if you're free around noon.\n\n- Mike",
        "snippet": "Want to grab lunch Friday at that new ramen place?",
        "category": "personal",
        "priority": 4,
        "confidence": 88.0,
        "status": "pending",
    },
    {
        "subject": "🎉 Flash Sale — 50% Off All Plans This Weekend Only!",
        "sender": "deals@saastools.com",
        "sender_name": "SaaS Tools",
        "body": "Don't miss out on our biggest sale of the year!\n\n50% OFF all annual plans this weekend only.\n\nUse code FLASH50 at checkout.\n\nShop now →",
        "snippet": "50% OFF all annual plans this weekend only. Use code FLASH50",
        "category": "marketing",
        "priority": 5,
        "confidence": 96.0,
        "status": "pending",
    },
    {
        "subject": "Re: Contract Review — NDA with Vertex Labs",
        "sender": "legal@acmecorp.com",
        "sender_name": "Diana Park",
        "body": "Hi,\n\nI've reviewed the NDA from Vertex Labs. A few concerns:\n\n1. Section 3.2 — non-compete clause is too broad (24 months)\n2. Section 5.1 — IP assignment language needs narrowing\n3. Governing law should be Delaware, not California\n\nPlease do not sign until we resolve these. Happy to schedule a call.\n\nDiana Park\nGeneral Counsel",
        "snippet": "I've reviewed the NDA from Vertex Labs. A few concerns...",
        "category": "risk",
        "priority": 2,
        "confidence": 89.0,
        "risk_score": 60.0,
        "status": "needs_review",
    },
    {
        "subject": "Sprint Planning — Monday 10 AM",
        "sender": "pm@acmecorp.com",
        "sender_name": "Alex Rivera",
        "body": "Reminder: Sprint planning meeting is Monday at 10 AM in Conference Room B.\n\nPlease come prepared with your task estimates for the auth migration project.\n\nAgenda:\n1. Previous sprint retro (15 min)\n2. Backlog grooming (20 min)\n3. Sprint commitment (25 min)",
        "snippet": "Sprint planning meeting is Monday at 10 AM. Please come prepared...",
        "category": "work",
        "priority": 3,
        "confidence": 92.0,
        "status": "pending",
    },
    {
        "subject": "Your AWS Bill for February 2026",
        "sender": "no-reply@aws.amazon.com",
        "sender_name": "Amazon Web Services",
        "body": "Your AWS bill for February 2026 is now available.\n\nTotal charges: $8,412.67\n\nTop services:\n- EC2: $4,200.00\n- RDS: $2,100.00\n- S3: $890.00\n- Other: $1,222.67\n\nView your bill in the AWS Console.",
        "snippet": "Your AWS bill for February 2026 is now available. Total: $8,412.67",
        "category": "billing",
        "priority": 2,
        "confidence": 95.0,
        "status": "pending",
    },
]

DEMO_TASKS = [
    {"email_idx": 0, "action": "flag", "reason": "Urgent board meeting deadline — needs department head sign-off by Thursday"},
    {"email_idx": 0, "action": "calendar", "reason": "Add board meeting reminder for Friday", "parameters": {"event": "Board Meeting — Q1 Review", "date": "2026-03-07"}},
    {"email_idx": 1, "action": "escalate", "reason": "Suspicious login from unrecognized location — requires immediate security review"},
    {"email_idx": 2, "action": "flag", "reason": "Invoice due March 15 — payment needed to avoid service disruption"},
    {"email_idx": 2, "action": "label", "reason": "Categorize as billing for tracking", "parameters": {"label": "billing/invoices"}},
    {"email_idx": 4, "action": "archive", "reason": "Promotional email — low priority, safe to archive"},
    {"email_idx": 5, "action": "escalate", "reason": "Legal review flagged contract issues — do not sign until resolved"},
    {"email_idx": 6, "action": "calendar", "reason": "Add sprint planning to calendar", "parameters": {"event": "Sprint Planning", "date": "2026-03-09", "time": "10:00"}},
]


class Command(BaseCommand):
    help = "Seed the database with demo emails and tasks"

    def handle(self, *args, **options):
        Email.objects.all().delete()
        Task.objects.all().delete()

        now = timezone.now()
        emails = []
        for i, data in enumerate(DEMO_EMAILS):
            data.setdefault("risk_score", 0.0)
            email = Email.objects.create(
                received_at=now - timedelta(hours=i * 3, minutes=i * 17),
                **data,
            )
            emails.append(email)
            self.stdout.write(f"  Created email: {email.subject[:50]}")

        for t in DEMO_TASKS:
            Task.objects.create(
                email=emails[t["email_idx"]],
                action=t["action"],
                reason=t["reason"],
                parameters=t.get("parameters", {}),
            )

        self.stdout.write(self.style.SUCCESS(
            f"\nSeeded {len(emails)} emails and {len(DEMO_TASKS)} tasks."
        ))
