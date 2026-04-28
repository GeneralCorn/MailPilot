from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mailpilot.settings")
django.setup()

from django.conf import settings as django_settings
django_settings.ALLOWED_HOSTS = ["*"]

import pytest
from django.test import Client

from triage import persistence
from triage.schemas import Action, Status


SAMPLE = [
    {"id": "e1", "subject": "Standup", "sender": "lead@acme.com", "thread_id": "t1"},
    {"id": "e2", "subject": "Other", "sender": "x@y.com", "thread_id": "t2"},
]


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    persistence.init_db(tmp_path / "db.sqlite3", migrate_from_json=False)
    with patch("triage.views._load", return_value=SAMPLE), \
         patch("triage.storage._load", return_value=SAMPLE):
        yield
    persistence.reset_conn()


def _seed_proposal(email_id="e1", tool="calendar", **params):
    persistence.update_email_state(email_id, status=Status.AWAITING_APPROVAL.value, proposed_actions=[
        {"tool": tool, "parameters": {"email_id": email_id, **params}, "reason": "x"}
    ])


def test_proposals_list_returns_pending(client_unused=None):
    _seed_proposal(tool="calendar", title="Sync", start_time="2026-05-01T10:00:00")
    resp = Client().get("/email/0/proposals/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["email_id"] == "e1"
    assert len(data["proposed_actions"]) == 1
    assert data["proposed_actions"][0]["tool"] == "calendar"


def test_approve_executes_tool_and_clears_proposal():
    _seed_proposal(tool="calendar", title="Sync", start_time="2026-05-01T10:00:00",
                   end_time="2026-05-01T10:30:00")
    fake_result = MagicMock(success=True, tool=Action.CALENDAR, message="event created", data={"calendar_event_id": "ev1"})

    with patch("triage.runtime.Runtime") as RT:
        RT.return_value.run_tool.return_value = fake_result
        resp = Client().post("/email/0/proposals/0/approve/")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tool"] == "calendar"
    assert body["new_status"] == Status.DONE.value
    # proposal removed
    assert persistence.get_email_state("e1")["proposed_actions"] == []
    # external_actions records the executed approval
    history = persistence.get_email_state("e1")["external_actions"]
    assert history[0]["tool"] == "calendar"
    assert history[0]["success"] is True


def test_approve_merges_user_edited_params():
    _seed_proposal(tool="calendar", title="Old title", start_time="t1", end_time="t2")
    captured = {}

    fake_result = MagicMock(success=True, tool=Action.CALENDAR, message="ok", data={})
    with patch("triage.runtime.Runtime") as RT:
        def capture(call):
            captured["call"] = call
            return fake_result
        RT.return_value.run_tool.side_effect = capture
        resp = Client().post("/email/0/proposals/0/approve/?title=New+Title&location=Room+4")

    assert resp.status_code == 200
    call = captured["call"]
    assert call.parameters["title"] == "New Title"
    assert call.parameters["location"] == "Room 4"
    assert call.parameters["start_time"] == "t1"  # untouched


def test_reject_drops_proposal_without_executing():
    _seed_proposal(tool="send_email", kind="rsvp", body="...")
    with patch("triage.runtime.Runtime") as RT:
        resp = Client().post("/email/0/proposals/0/reject/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rejected_tool"] == "send_email"
    assert body["new_status"] == Status.DONE.value  # nothing else pending, no external history
    assert persistence.get_email_state("e1")["proposed_actions"] == []
    RT.return_value.run_tool.assert_not_called()


def test_approve_out_of_range_returns_404():
    resp = Client().post("/email/0/proposals/5/approve/")
    assert resp.status_code == 404


def test_status_stays_awaiting_when_other_proposals_remain():
    persistence.update_email_state("e1", status=Status.AWAITING_APPROVAL.value, proposed_actions=[
        {"tool": "calendar", "parameters": {"email_id": "e1", "title": "A"}},
        {"tool": "send_email", "parameters": {"email_id": "e1", "kind": "rsvp", "body": "ok"}},
    ])
    resp = Client().post("/email/0/proposals/0/reject/")
    assert resp.json()["new_status"] == Status.AWAITING_APPROVAL.value
    assert len(persistence.get_email_state("e1")["proposed_actions"]) == 1


def test_pipeline_skips_email_with_pending_proposals():
    """run_pipeline should treat an email with existing proposals as already-being-handled."""
    from triage import pipeline as pipe
    from triage.schemas import Category, Message
    from triage.agent.evaluator import EvaluatorResult

    _seed_proposal(tool="calendar", title="Sync")

    seen_in_loop: list[str] = []

    def fake_route(email, state, *, feedback=None):
        seen_in_loop.append(email.id)
        state.classifications[email.id] = Category.WORK
        state.confidence_scores[email.id] = 0.9

    def fake_evaluate(email, state):
        state.risk_scores[email.id] = 0.05
        return EvaluatorResult(
            final_category=Category.WORK, confidence=0.9, risk_score=0.05,
            needs_review=False, explanation="ok",
        )

    def fake_rank(state):
        for m in state.messages:
            state.priority_queue.append((m.id, 0.5))

    def fake_work(email, state, runtime=None):
        state.email_status[email.id] = Status.DONE
        return Status.DONE

    emails = [Message(id="e1", subject="x", sender="y"), Message(id="e2", subject="x", sender="y")]
    with patch("triage.agent.loop.route", side_effect=fake_route), \
         patch("triage.agent.loop.evaluate", side_effect=fake_evaluate), \
         patch("triage.pipeline.rank", side_effect=fake_rank), \
         patch("triage.pipeline.work", side_effect=fake_work):
        pipe.run_pipeline(emails)

    assert seen_in_loop == ["e2"]  # e1 was skipped at input dedupe
