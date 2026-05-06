from __future__ import annotations

import re

from triage.schemas import Message

# Lookalike / suspicious sender patterns (linkedln, paypa1, etc.)
_RISK_DOMAIN_PATTERNS = [
    r"linkedln", r"paypa1", r"-secure\.", r"verify-", r"bank-verify",
    r"-confirm\.", r"\.tk$", r"\.ru$",
]
_RISK_KEYWORDS = re.compile(
    r"\b(verify your|account.{0,20}lock|suspicious sign-in|urgent.{0,20}action|"
    r"click.{0,15}here.{0,15}now|wire transfer|bitcoin|crypto)\b",
    re.IGNORECASE,
)
_MARKETING_KEYWORDS = re.compile(
    r"\b(unsubscribe|% off|sale|deal|newsletter|promo|webinar|exclusive offer)\b",
    re.IGNORECASE,
)
_BILLING_KEYWORDS = re.compile(
    r"\b(invoice|receipt|payment|charged|amount due|paid|statement|subscription)\b",
    re.IGNORECASE,
)
_WORK_KEYWORDS = re.compile(
    r"\b(meeting|review|deadline|standup|sprint|kickoff|onboarding|status update|"
    r"please confirm|rsvp)\b",
    re.IGNORECASE,
)
_HIGH_URGENCY = re.compile(r"\b(urgent|asap|today|eod|deadline|overdue|past due)\b", re.IGNORECASE)

_PERSONAL_DOMAINS = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com")


def _classify(msg: Message) -> str:
    text = f"{msg.subject} {msg.body_plain or msg.snippet}"
    sender = msg.sender.lower()

    if any(re.search(p, sender) for p in _RISK_DOMAIN_PATTERNS) or _RISK_KEYWORDS.search(text):
        return "risk"
    if _MARKETING_KEYWORDS.search(text):
        return "marketing"
    if _BILLING_KEYWORDS.search(text):
        return "billing"
    if _WORK_KEYWORDS.search(text):
        return "work"
    if any(d in sender for d in _PERSONAL_DOMAINS):
        return "personal"
    return "work"


def _priority(msg: Message, category: str) -> str:
    text = f"{msg.subject} {msg.body_plain or msg.snippet}"
    if category == "risk":
        return "high"
    if _HIGH_URGENCY.search(text):
        return "high"
    if category in ("work", "billing"):
        return "medium"
    return "low"


def _action(category: str) -> str:
    # Limited to gmail-style filter actions; no calendar / reply_draft capability.
    return {
        "marketing": "archive",
        "risk":      "flag",
        "billing":   "flag",
        "work":      "label",
        "personal":  "no_action",
    }.get(category, "no_action")


def run(emails: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in emails:
        cat = _classify(m)
        out.append({
            "id": m.id,
            "category": cat,
            "priority": _priority(m, cat),
            "action": _action(cat),
            "needs_review": cat == "risk",
            "proposed_actions": [],
            "external_actions": [],
        })
    return out
