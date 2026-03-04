import json
from unittest.mock import patch

import pytest

from triage.schemas import Action, Status, ToolResult
from triage.tools.tool_registry import (
    archive_email,
    batch_label,
    create_calendar_event,
    flag_email,
    label_email,
)

SAMPLE_EMAILS = [
    {"id": "e1", "subject": "Hello", "category": "unclassified", "flagged": False},
    {"id": "e2", "subject": "Meeting", "category": "unclassified", "flagged": False},
    {"id": "e3", "subject": "Invoice", "category": "unclassified", "flagged": False},
]


def _fresh_emails():
    return json.loads(json.dumps(SAMPLE_EMAILS))


@pytest.fixture(autouse=True)
def mock_storage():
    """Patch _load/_save so tests never touch the real JSON file."""
    store: list[dict] = _fresh_emails()

    def fake_load():
        return store

    def fake_save(emails):
        pass   

    with (
        patch("triage.tools.tool_registry._load", side_effect=fake_load),
        patch("triage.tools.tool_registry._save", side_effect=fake_save),
    ):
        yield store


def test_label_email_assigns_category(mock_storage):
    result = label_email("e1", "work")

    assert isinstance(result, ToolResult)
    assert result.tool == Action.LABEL
    assert result.success is True
    assert result.data == {"category": "work"}
    assert mock_storage[0]["category"] == "work"

def test_flag_email_sets_flagged(mock_storage):
    result = flag_email("e1", flag=True)

    assert result.tool == Action.FLAG
    assert result.success is True
    assert result.data == {"flagged": True}
    assert mock_storage[0]["flagged"] is True


def test_flag_email_unsets_flagged(mock_storage):
    mock_storage[0]["flagged"] = True
    result = flag_email("e1", flag=False)

    assert result.data == {"flagged": False}
    assert mock_storage[0]["flagged"] is False

def test_archive_email_moves_to_folder_and_sets_done(mock_storage):
    result = archive_email("e1", folder="trash")

    assert result.tool == Action.ARCHIVE
    assert result.success is True
    assert result.data == {"folder": "trash"}
    assert mock_storage[0]["folder"] == "trash"
    assert mock_storage[0]["status"] == Status.DONE.value


def test_archive_email_defaults_to_archive_folder(mock_storage):
    result = archive_email("e2")

    assert result.data == {"folder": "archive"}
    assert mock_storage[1]["folder"] == "archive"


def test_create_calendar_event_adds_event_and_sets_done(mock_storage):
    result = create_calendar_event(
        email_id="e2",
        title="Standup",
        start_time="2026-03-05T09:00:00Z",
        end_time="2026-03-05T09:30:00Z",
        location="Room 4",
        response="accept",
    )

    assert result.tool == Action.CALENDAR
    assert result.success is True
    assert result.data["title"] == "Standup"
    assert result.data["start_time"] == "2026-03-05T09:00:00Z"
    assert result.data["end_time"] == "2026-03-05T09:30:00Z"
    assert result.data["location"] == "Room 4"
    assert result.data["response"] == "accept"
    assert mock_storage[1]["status"] == Status.DONE.value
    assert mock_storage[1]["calendar_event"]["title"] == "Standup"


def test_batch_label_applies_category_to_multiple_emails(mock_storage):
    results = batch_label(["e1", "e2", "e3"], "billing")

    assert len(results) == 3
    assert all(isinstance(r, ToolResult) for r in results)
    assert all(r.tool == Action.LABEL for r in results)
    assert all(r.success is True for r in results)
    for email in mock_storage:
        assert email["category"] == "billing"
