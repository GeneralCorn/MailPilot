from __future__ import annotations

from triage.schemas import Message

from benchmarks.baselines import rule_based


def _msg(eid: str, subject: str, sender: str = "x@y.com", body: str = "") -> Message:
    return Message(id=eid, subject=subject, sender=sender, body_plain=body)


def test_phishing_classified_as_risk():
    out = rule_based.run([_msg("a", "URGENT: verify your account",
                                sender="security@linkedln-secure.com")])
    assert out[0]["category"] == "risk"
    assert out[0]["needs_review"] is True
    assert out[0]["action"] == "flag"
    assert out[0]["priority"] == "high"


def test_marketing_archives():
    out = rule_based.run([_msg("a", "30% off everything", sender="deals@shop.com",
                                body="Unsubscribe at the bottom")])
    assert out[0]["category"] == "marketing"
    assert out[0]["action"] == "archive"
    assert out[0]["priority"] == "low"


def test_billing_flags():
    out = rule_based.run([_msg("a", "Invoice 4421", sender="billing@vendor.com",
                                body="Payment received")])
    assert out[0]["category"] == "billing"
    assert out[0]["action"] == "flag"


def test_work_with_urgency_high_priority():
    out = rule_based.run([_msg("a", "RSVP today: Q3 review",
                                sender="manager@acme.com")])
    assert out[0]["category"] == "work"
    assert out[0]["priority"] == "high"


def test_personal_email_no_action():
    out = rule_based.run([_msg("a", "Coffee Friday?",
                                sender="alex@gmail.com",
                                body="Want to grab coffee?")])
    assert out[0]["category"] == "personal"
    assert out[0]["action"] == "no_action"


def test_schema_shape():
    out = rule_based.run([_msg("a", "anything")])
    for k in ("id", "category", "priority", "action", "needs_review"):
        assert k in out[0]
