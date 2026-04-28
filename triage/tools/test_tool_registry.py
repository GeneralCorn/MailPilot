from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from triage import persistence
from triage.schemas import Action, Status, ToolResult
from triage.tools.actions import (
    archive_email,
    batch_label,
    create_calendar_event,
    draft_reply,
    flag_email,
    label_email,
    send_confirmation_email,
    send_rsvp_email,
    summarize_email,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    persistence.init_db(tmp_path / "db.sqlite3", migrate_from_json=False)
    with patch(
        "triage.storage.get_email_body",
        return_value={"subject": "stub", "sender": "alice@example.com", "thread_id": "t1"},
    ):
        yield
    persistence.reset_conn()


def test_label_email_assigns_category():
    result = label_email("e1", "work")

    assert isinstance(result, ToolResult)
    assert result.tool == Action.LABEL
    assert result.success is True
    assert result.data == {"category": "work"}
    assert persistence.get_email_state("e1")["category"] == "work"


def test_flag_email_sets_flagged():
    result = flag_email("e1", flag=True)

    assert result.tool == Action.FLAG
    assert result.data == {"flagged": True}
    assert persistence.get_email_state("e1")["flagged"] is True


def test_flag_email_unsets_flagged():
    flag_email("e1", flag=True)
    result = flag_email("e1", flag=False)

    assert result.data == {"flagged": False}
    assert persistence.get_email_state("e1")["flagged"] is False


def test_archive_email_moves_to_folder_and_sets_done():
    result = archive_email("e1", folder="trash")

    assert result.tool == Action.ARCHIVE
    assert result.data == {"folder": "trash"}
    state = persistence.get_email_state("e1")
    assert state["folder"] == "trash"
    assert state["status"] == Status.DONE.value


def test_archive_email_defaults_to_archive_folder():
    result = archive_email("e2")

    assert result.data == {"folder": "archive"}
    assert persistence.get_email_state("e2")["folder"] == "archive"


def test_create_calendar_event_mock_path_skips_api():
    with patch("triage.gmail.get_calendar_service") as svc:
        result = create_calendar_event(
            email_id="e2",
            title="Standup",
            start_time="2026-03-05T09:00:00Z",
            end_time="2026-03-05T09:30:00Z",
            location="Room 4",
            response="accept",
        )
    svc.assert_not_called()
    assert result.tool == Action.CALENDAR
    assert result.data["title"] == "Standup"
    state = persistence.get_email_state("e2")
    assert state["status"] == Status.DONE.value
    assert state["calendar_event"]["title"] == "Standup"


def test_create_calendar_event_real_path_calls_api(monkeypatch):
    monkeypatch.setenv("MAILPILOT_MOCK_TOOLS", "0")
    fake_svc = MagicMock()
    fake_svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "evt-123",
        "htmlLink": "https://calendar.google.com/evt-123",
    }
    with patch("triage.gmail.get_calendar_service", return_value=fake_svc):
        result = create_calendar_event(
            email_id="e3",
            title="Sync",
            start_time="2026-04-29T10:00:00Z",
            end_time="2026-04-29T10:30:00Z",
        )

    fake_svc.events.return_value.insert.assert_called_once()
    insert_kwargs = fake_svc.events.return_value.insert.call_args.kwargs
    assert insert_kwargs["calendarId"] == "primary"
    assert insert_kwargs["body"]["summary"] == "Sync"
    assert result.data["calendar_event_id"] == "evt-123"
    assert persistence.get_email_state("e3")["calendar_event"]["calendar_event_id"] == "evt-123"


def test_draft_reply_real_path_creates_gmail_draft(monkeypatch):
    monkeypatch.setenv("MAILPILOT_MOCK_TOOLS", "0")
    fake_svc = MagicMock()
    fake_svc.users.return_value.drafts.return_value.create.return_value.execute.return_value = {"id": "draft-7"}
    with patch("triage.gmail.get_gmail_service", return_value=fake_svc):
        result = draft_reply(email_id="e4", body="Sounds good.")

    fake_svc.users.return_value.drafts.return_value.create.assert_called_once()
    assert result.data["gmail_draft_id"] == "draft-7"
    assert persistence.get_email_state("e4")["draft_reply"]["gmail_draft_id"] == "draft-7"


def test_summarize_email_writes_summary_to_state():
    result = summarize_email("e5", "Project kickoff next Tuesday at 3 PM in Room A.")

    assert result.tool == Action.SUMMARIZE
    assert result.success is True
    assert persistence.get_email_state("e5")["summary"].startswith("Project kickoff")


def test_send_rsvp_email_mock_path_records_external_action():
    result = send_rsvp_email("e6", decision="accept", body="See you there.")

    assert result.tool == Action.SEND_EMAIL
    assert result.success is True
    actions = persistence.get_email_state("e6")["external_actions"]
    assert len(actions) == 1
    assert actions[0]["kind"] == "rsvp:accept"
    assert actions[0]["mock"] is True
    assert actions[0]["to"] == "alice@example.com"


def test_send_rsvp_email_idempotent_skip_on_repeat():
    send_rsvp_email("e7", decision="accept", body="ok")
    second = send_rsvp_email("e7", decision="accept", body="ok again")

    assert second.data.get("skipped") is True
    actions = persistence.get_email_state("e7")["external_actions"]
    assert len(actions) == 1


def test_send_rsvp_email_different_decision_is_not_idempotent():
    send_rsvp_email("e8", decision="accept", body="ok")
    second = send_rsvp_email("e8", decision="decline", body="cannot make it")

    assert second.data.get("skipped") is None
    actions = persistence.get_email_state("e8")["external_actions"]
    assert {a["kind"] for a in actions} == {"rsvp:accept", "rsvp:decline"}


def test_send_confirmation_email_real_path_calls_send(monkeypatch):
    monkeypatch.setenv("MAILPILOT_MOCK_TOOLS", "0")
    with patch("triage.gmail.send_email", return_value={"id": "msg-99"}) as send:
        result = send_confirmation_email("e9", body="Confirmed.")

    send.assert_called_once()
    args, kwargs = send.call_args
    assert args[0] == "alice@example.com"
    assert kwargs.get("thread_id") == "t1"
    actions = persistence.get_email_state("e9")["external_actions"]
    assert actions[0]["gmail_message_id"] == "msg-99"


def test_batch_label_applies_category_to_multiple_emails():
    results = batch_label(["e10", "e11", "e12"], "billing")

    assert len(results) == 3
    assert all(r.tool == Action.LABEL and r.success for r in results)
    for eid in ("e10", "e11", "e12"):
        assert persistence.get_email_state(eid)["category"] == "billing"
