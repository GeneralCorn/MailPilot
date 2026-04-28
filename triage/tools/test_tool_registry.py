from pathlib import Path
from unittest.mock import patch

import pytest

from triage import persistence
from triage.schemas import Action, Status, ToolResult
from triage.tools.actions import (
    archive_email,
    batch_label,
    create_calendar_event,
    flag_email,
    label_email,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    persistence.init_db(tmp_path / "db.sqlite3", migrate_from_json=False)
    # draft_reply does a body lookup against emails.json — keep it isolated from real data
    with patch("triage.storage.get_email_body", return_value={"subject": "stub"}):
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


def test_create_calendar_event_adds_event_and_sets_done():
    result = create_calendar_event(
        email_id="e2",
        title="Standup",
        start_time="2026-03-05T09:00:00Z",
        end_time="2026-03-05T09:30:00Z",
        location="Room 4",
        response="accept",
    )

    assert result.tool == Action.CALENDAR
    assert result.data["title"] == "Standup"
    assert result.data["start_time"] == "2026-03-05T09:00:00Z"
    assert result.data["location"] == "Room 4"
    state = persistence.get_email_state("e2")
    assert state["status"] == Status.DONE.value
    assert state["calendar_event"]["title"] == "Standup"


def test_batch_label_applies_category_to_multiple_emails():
    results = batch_label(["e1", "e2", "e3"], "billing")

    assert len(results) == 3
    assert all(r.tool == Action.LABEL and r.success for r in results)
    for eid in ("e1", "e2", "e3"):
        assert persistence.get_email_state(eid)["category"] == "billing"
