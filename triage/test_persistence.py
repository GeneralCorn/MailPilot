from __future__ import annotations

from pathlib import Path

import pytest

from triage import persistence
from triage.agent import wrapper
from triage.schemas import Category, Message, State, Status


@pytest.fixture
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    db = tmp_path / "mailpilot.sqlite3"
    persistence.init_db(db)
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
