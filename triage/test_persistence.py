from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from triage import persistence
from triage.agent import wrapper
from triage.schemas import Category, Message, State, Status


@pytest.fixture
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    db = tmp_path / "mailpilot.sqlite3"
    persistence.init_db(db, migrate_from_json=False)
    yield db
    persistence.reset_conn()


def _state_with(email_id: str, category: Category) -> State:
    s = State(messages=[Message(id=email_id, subject="x", sender="a@b")])
    s.classifications[email_id] = category
    return s


def test_snapshot_and_latest_roundtrip(tmp_db):
    run_id = "r1"
    persistence.start_run(run_id)
    s = _state_with("e1", Category.WORK)
    persistence.snapshot(run_id, "e1", "router", s)

    found = persistence.latest_snapshot(run_id, "e1")
    assert found is not None
    stage, restored = found
    assert stage == "router"
    assert restored.classifications["e1"] == Category.WORK


def test_latest_snapshot_returns_most_recent(tmp_db):
    run_id = "r1"
    persistence.start_run(run_id)
    s1 = _state_with("e1", Category.WORK)
    persistence.snapshot(run_id, "e1", "router", s1)
    s2 = _state_with("e1", Category.PERSONAL)
    persistence.snapshot(run_id, "e1", "evaluator", s2)

    stage, restored = persistence.latest_snapshot(run_id, "e1")
    assert stage == "evaluator"
    assert restored.classifications["e1"] == Category.PERSONAL


def test_mark_processed_dedupe(tmp_db):
    persistence.start_run("r1")
    assert not persistence.is_processed("e1")
    persistence.mark_processed("e1", "r1", Status.DONE)
    assert persistence.is_processed("e1")

    persistence.mark_processed("e2", "r1", Status.FLAGGED)
    assert not persistence.is_processed("e2")  # only DONE counts as processed


def test_run_stage_writes_snapshot_after_fn(tmp_db):
    persistence.start_run("r1")

    def stage_fn(state: State) -> None:
        state.classifications["e1"] = Category.BILLING

    state = State(messages=[Message(id="e1", subject="x", sender="a@b")])
    wrapper.run_stage("router", stage_fn, state, run_id="r1", email_id="e1")

    stage, restored = persistence.latest_snapshot("r1", "e1")
    assert stage == "router"
    assert restored.classifications["e1"] == Category.BILLING


def test_run_stage_snapshots_error_and_reraises(tmp_db):
    persistence.start_run("r1")

    def boom(state: State) -> None:
        state.classifications["e1"] = Category.RISK
        raise RuntimeError("nope")

    state = State(messages=[Message(id="e1", subject="x", sender="a@b")])
    with pytest.raises(RuntimeError):
        wrapper.run_stage("router", boom, state, run_id="r1", email_id="e1")

    stage, restored = persistence.latest_snapshot("r1", "e1")
    assert stage == "router:error"
    assert restored.classifications["e1"] == Category.RISK


def test_resume_stage_index(tmp_db):
    persistence.start_run("r1")
    state = State(messages=[Message(id="e1", subject="x", sender="a@b")])
    wrapper.run_stage("router", lambda s: None, state, run_id="r1", email_id="e1")
    wrapper.run_stage("evaluator", lambda s: None, state, run_id="r1", email_id="e1")

    stages = ("input", "router", "evaluator", "ranker", "worker")
    assert wrapper.resume_stage_index("r1", "e1", stages) == 3
    assert wrapper.resume_stage_index("r1", "e_unknown", stages) == 0


def test_email_state_defaults_for_unknown_id(tmp_db):
    s = persistence.get_email_state("nope")
    assert s["category"] == "unclassified"
    assert s["flagged"] is False
    assert s["escalations"] == []
    assert s["draft_reply"] is None


def test_email_state_roundtrip_scalar_and_json(tmp_db):
    persistence.update_email_state("e1", category="work", flagged=True, status="done")
    persistence.update_email_state("e1", calendar_event={"title": "Standup"})

    s = persistence.get_email_state("e1")
    assert s["category"] == "work"
    assert s["flagged"] is True
    assert s["status"] == "done"
    assert s["calendar_event"] == {"title": "Standup"}


def test_email_state_partial_update_does_not_clobber_other_fields(tmp_db):
    persistence.update_email_state("e1", category="work", flagged=True)
    persistence.update_email_state("e1", status="done")

    s = persistence.get_email_state("e1")
    assert s["category"] == "work"
    assert s["flagged"] is True
    assert s["status"] == "done"


def test_append_email_list_field_does_not_overwrite(tmp_db):
    persistence.append_email_list_field("e1", "notes", {"text": "first"})
    persistence.append_email_list_field("e1", "notes", {"text": "second"})

    notes = persistence.get_email_state("e1")["notes"]
    assert [n["text"] for n in notes] == ["first", "second"]


def test_append_email_list_field_rejects_unknown_field(tmp_db):
    with pytest.raises(ValueError):
        persistence.append_email_list_field("e1", "category", "x")


def test_update_email_state_rejects_unknown_field(tmp_db):
    with pytest.raises(ValueError):
        persistence.update_email_state("e1", bogus="x")


def test_list_email_states_returns_only_requested_ids(tmp_db):
    persistence.update_email_state("e1", category="work")
    persistence.update_email_state("e2", category="personal")
    persistence.update_email_state("e3", category="risk")

    out = persistence.list_email_states(["e1", "e3"])
    assert set(out.keys()) == {"e1", "e3"}
    assert out["e1"]["category"] == "work"
    assert out["e3"]["category"] == "risk"


def test_migrate_emails_json_to_state(tmp_path: Path, tmp_db):
    fake_emails = [
        {"id": "e1", "subject": "hi", "category": "work", "flagged": True, "status": "done"},
        {"id": "e2", "subject": "yo", "category": "personal",
         "calendar_event": {"title": "Standup"}, "notes": [{"text": "hey"}]},
    ]
    with patch("triage.storage._load", return_value=fake_emails):
        written = persistence.migrate_emails_json_to_state()
    assert written == 2
    assert persistence.get_email_state("e1")["category"] == "work"
    assert persistence.get_email_state("e1")["flagged"] is True
    assert persistence.get_email_state("e2")["calendar_event"]["title"] == "Standup"
    assert persistence.get_email_state("e2")["notes"] == [{"text": "hey"}]


def test_migrate_emails_json_to_state_is_idempotent(tmp_path: Path, tmp_db):
    fake_emails = [{"id": "e1", "subject": "x", "category": "work"}]
    with patch("triage.storage._load", return_value=fake_emails):
        first = persistence.migrate_emails_json_to_state()
        second = persistence.migrate_emails_json_to_state()
    assert first == 1
    assert second == 0  # second call must not re-write


def test_proposed_actions_roundtrip_and_helpers(tmp_db):
    persistence.update_email_state(
        "e1",
        proposed_actions=[
            {"tool": "calendar", "parameters": {"title": "A", "start_time": "t1"}},
            {"tool": "send_email", "parameters": {"kind": "rsvp", "decision": "accept", "body": "ok"}},
        ],
    )
    assert persistence.has_pending_proposals("e1") is True
    assert not persistence.has_pending_proposals("e2")

    updated = persistence.update_proposed_action("e1", 0, {"start_time": "t2", "location": "Room 4"})
    assert updated["parameters"]["start_time"] == "t2"
    assert updated["parameters"]["location"] == "Room 4"
    assert updated["parameters"]["title"] == "A"  # original key preserved

    removed = persistence.remove_proposed_action("e1", 0)
    assert removed["tool"] == "calendar"
    rest = persistence.get_email_state("e1")["proposed_actions"]
    assert len(rest) == 1
    assert rest[0]["tool"] == "send_email"

    # remove last → should clear
    persistence.remove_proposed_action("e1", 0)
    assert persistence.has_pending_proposals("e1") is False


def test_proposed_action_helpers_handle_out_of_range(tmp_db):
    assert persistence.remove_proposed_action("nope", 0) is None
    assert persistence.update_proposed_action("nope", 0, {"x": 1}) is None


def test_init_db_is_idempotent_with_added_columns(tmp_path: Path):
    persistence.reset_conn()
    db = tmp_path / "db.sqlite3"
    persistence.init_db(db, migrate_from_json=False)
    persistence.init_db(db, migrate_from_json=False)  # second call must not raise
    persistence.update_email_state("e1", proposed_actions=[{"tool": "calendar"}])
    assert persistence.has_pending_proposals("e1")
    persistence.reset_conn()
