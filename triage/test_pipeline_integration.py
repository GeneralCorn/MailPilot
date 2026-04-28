from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from triage import persistence
from triage import pipeline as pipe
from triage.agent.evaluator import EvaluatorResult
from triage.schemas import Category, Message, Priority, State, Status


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    persistence.reset_conn()
    persistence.init_db(tmp_path / "db.sqlite3", migrate_from_json=False)
    yield
    persistence.reset_conn()


# ── shared fakes ─────────────────────────────────────────────────────────────

_CATS = {"e1": Category.WORK, "e2": Category.MARKETING, "e3": Category.RISK}
_RISKS = {"e1": 0.05, "e2": 0.02, "e3": 0.95}


def _fake_route(email, state, *, feedback=None):
    state.classifications[email.id] = _CATS.get(email.id, Category.UNCLASSIFIED)
    state.confidence_scores[email.id] = 0.9


def _fake_evaluate(email, state):
    risk = _RISKS.get(email.id, 0.05)
    state.risk_scores[email.id] = risk
    return EvaluatorResult(
        final_category=state.classifications[email.id],
        confidence=0.9,
        risk_score=risk,
        needs_review=email.id == "e3",
        explanation="ok",
    )


def _fake_rank(state):
    ordered = sorted(state.messages, key=lambda m: m.id)
    state.priority_queue = [(m.id, 1.0 - i * 0.3) for i, m in enumerate(ordered)]
    for m in ordered:
        state.priorities[m.id] = Priority.NORMAL


def _fake_work(email, state, runtime=None):
    state.email_status[email.id] = Status.DONE
    return Status.DONE


def _patch_all_agents(*, route_fn=None, evaluate_fn=None, rank_fn=None, work_fn=None):
    return [
        patch("triage.agent.loop.route", side_effect=route_fn or _fake_route),
        patch("triage.agent.loop.evaluate", side_effect=evaluate_fn or _fake_evaluate),
        patch("triage.pipeline.rank", side_effect=rank_fn or _fake_rank),
        patch("triage.pipeline.work", side_effect=work_fn or _fake_work),
    ]


# ── tests ────────────────────────────────────────────────────────────────────

def test_run_pipeline_end_to_end_three_emails():
    emails = [
        Message(id="e1", subject="Standup", sender="lead@acme.com"),
        Message(id="e2", subject="Sale 50% off", sender="deals@store.com"),
        Message(id="e3", subject="Verify your account", sender="bank@x.ru"),
    ]

    patches = _patch_all_agents()
    for p in patches:
        p.start()
    try:
        state, run_id = pipe.run_pipeline(emails)
    finally:
        for p in patches:
            p.stop()

    assert run_id.startswith("run-")
    assert set(state.classifications.keys()) == {"e1", "e2", "e3"}
    assert set(state.email_status.keys()) == {"e1", "e2", "e3"}
    assert all(s == Status.DONE for s in state.email_status.values())

    conn = persistence.get_conn()
    eval_snaps = conn.execute(
        "SELECT COUNT(*) FROM email_snapshots WHERE run_id=? AND stage='evaluator'",
        (run_id,),
    ).fetchone()[0]
    ranker_snaps = conn.execute(
        "SELECT COUNT(*) FROM email_snapshots WHERE run_id=? AND stage='ranker' AND email_id IS NULL",
        (run_id,),
    ).fetchone()[0]
    worker_snaps = conn.execute(
        "SELECT COUNT(*) FROM email_snapshots WHERE run_id=? AND stage='worker'",
        (run_id,),
    ).fetchone()[0]
    assert eval_snaps == 3
    assert ranker_snaps == 1
    assert worker_snaps == 3

    for eid in ("e1", "e2", "e3"):
        assert persistence.is_processed(eid)
        assert persistence.get_email_state(eid)["status"] == Status.DONE.value


def test_input_dedupe_skips_already_processed_emails():
    persistence.start_run("r0")
    persistence.mark_processed("e1", "r0", Status.DONE)

    emails = [
        Message(id="e1", subject="x", sender="y"),
        Message(id="e2", subject="x", sender="y"),
    ]

    seen: list[str] = []

    def track_route(email, state, *, feedback=None):
        seen.append(email.id)
        _fake_route(email, state)

    patches = _patch_all_agents(route_fn=track_route)
    for p in patches:
        p.start()
    try:
        state, _ = pipe.run_pipeline(emails)
    finally:
        for p in patches:
            p.stop()

    assert seen == ["e2"]
    assert "e1" not in state.email_status
    assert "e2" in state.email_status


def test_resume_skips_completed_evaluator_stage():
    pre = State(messages=[Message(id="e1", subject="x", sender="y")])
    pre.classifications["e1"] = Category.WORK
    pre.confidence_scores["e1"] = 0.9
    pre.risk_scores["e1"] = 0.05
    persistence.start_run("r1")
    persistence.snapshot("r1", "e1", "evaluator", pre)

    emails = [Message(id="e1", subject="x", sender="y")]

    route_calls: list[str] = []

    def track_route(email, state, *, feedback=None):
        route_calls.append(email.id)
        _fake_route(email, state)

    patches = _patch_all_agents(route_fn=track_route)
    for p in patches:
        p.start()
    try:
        state, run_id = pipe.run_pipeline(emails, run_id="r1")
    finally:
        for p in patches:
            p.stop()

    assert run_id == "r1"
    assert route_calls == []  # evaluator stage was already done
    assert state.classifications["e1"] == Category.WORK  # restored from snapshot
    assert state.email_status["e1"] == Status.DONE


def test_resume_skips_already_finished_worker():
    pre = State(messages=[Message(id="e1", subject="x", sender="y")])
    pre.classifications["e1"] = Category.WORK
    pre.email_status["e1"] = Status.DONE
    persistence.start_run("r2")
    persistence.snapshot("r2", "e1", "evaluator", pre)
    persistence.snapshot("r2", None, "ranker", pre)
    persistence.snapshot("r2", "e1", "worker", pre)

    work_calls: list[str] = []

    def track_work(email, state, runtime=None):
        work_calls.append(email.id)
        return Status.DONE

    emails = [Message(id="e1", subject="x", sender="y")]
    patches = _patch_all_agents(work_fn=track_work)
    for p in patches:
        p.start()
    try:
        pipe.run_pipeline(emails, run_id="r2")
    finally:
        for p in patches:
            p.stop()

    assert work_calls == []  # worker already had a snapshot for e1


def test_run_pipeline_marks_run_finished():
    emails = [Message(id="e1", subject="x", sender="y")]
    patches = _patch_all_agents()
    for p in patches:
        p.start()
    try:
        _, run_id = pipe.run_pipeline(emails)
    finally:
        for p in patches:
            p.stop()

    conn = persistence.get_conn()
    row = conn.execute(
        "SELECT finished_at, status FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert row[1] == "done"
