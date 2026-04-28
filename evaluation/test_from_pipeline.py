from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation import from_pipeline
from evaluation.runner import EvaluatorConfig, evaluate
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


def _fake_route(email, state, *, feedback=None):
    cat_map = {"e1": Category.WORK, "e2": Category.MARKETING, "e3": Category.RISK}
    state.classifications[email.id] = cat_map.get(email.id, Category.UNCLASSIFIED)
    state.confidence_scores[email.id] = 0.9


def _fake_evaluate(email, state):
    risk_map = {"e1": 0.05, "e2": 0.02, "e3": 0.95}
    risk = risk_map.get(email.id, 0.05)
    state.risk_scores[email.id] = risk
    return EvaluatorResult(
        final_category=state.classifications[email.id],
        confidence=0.9,
        risk_score=risk,
        needs_review=email.id == "e3",
        explanation="ok",
    )


def _fake_rank(state):
    state.priority_queue = [(m.id, 0.5) for m in state.messages]
    for m in state.messages:
        state.priorities[m.id] = Priority.NORMAL


def _fake_work(email, state, runtime=None):
    state.email_status[email.id] = Status.DONE
    return Status.DONE


def _run_pipeline_with_fakes(emails):
    patches = [
        patch("triage.agent.loop.route", side_effect=_fake_route),
        patch("triage.agent.loop.evaluate", side_effect=_fake_evaluate),
        patch("triage.pipeline.rank", side_effect=_fake_rank),
        patch("triage.pipeline.work", side_effect=_fake_work),
    ]
    for p in patches:
        p.start()
    try:
        return pipe.run_pipeline(emails)
    finally:
        for p in patches:
            p.stop()


def test_dump_uses_latest_run_when_run_id_omitted():
    emails = [Message(id="e1", subject="x", sender="y")]
    _, run_id = _run_pipeline_with_fakes(emails)

    items = from_pipeline.dump_run_to_agent_outputs()
    assert len(items) == 1
    assert items[0]["id"] == "e1"


def test_dump_writes_to_file():
    emails = [Message(id="e1", subject="x", sender="y")]
    _, run_id = _run_pipeline_with_fakes(emails)

    out_file = Path(persistence.DB_PATH).parent / "dump.json"
    items = from_pipeline.dump_run_to_agent_outputs(run_id, out_file)

    on_disk = json.loads(out_file.read_text())
    assert on_disk == items


def test_dump_shape_matches_runner_input():
    emails = [
        Message(id="e1", subject="meet", sender="a@b"),
        Message(id="e2", subject="sale", sender="c@d"),
    ]
    _, run_id = _run_pipeline_with_fakes(emails)

    items = from_pipeline.dump_run_to_agent_outputs(run_id)
    by_id = {it["id"]: it for it in items}

    assert by_id["e1"]["router_output"]["category"] == "work"
    assert by_id["e2"]["router_output"]["category"] == "marketing"
    assert by_id["e1"]["evaluator_output"]["final_category"] == "work"
    assert by_id["e1"]["evaluator_output"]["risk_score"] == 0.05
    assert by_id["e1"]["evaluator_output"]["needs_review"] is False


def test_dump_preserves_router_output_when_evaluator_overrides():
    """Router says marketing, evaluator overrides to risk — dump must keep both."""
    emails = [Message(id="e1", subject="x", sender="y")]

    def router_says_marketing(email, state, *, feedback=None):
        state.classifications[email.id] = Category.MARKETING
        state.confidence_scores[email.id] = 0.6

    def evaluator_overrides_to_risk(email, state):
        state.classifications[email.id] = Category.RISK
        state.confidence_scores[email.id] = 0.92
        state.risk_scores[email.id] = 0.95
        return EvaluatorResult(
            final_category=Category.RISK,
            confidence=0.92,
            risk_score=0.95,
            needs_review=True,
            explanation="phishing",
        )

    patches = [
        patch("triage.agent.loop.route", side_effect=router_says_marketing),
        patch("triage.agent.loop.evaluate", side_effect=evaluator_overrides_to_risk),
        patch("triage.pipeline.rank", side_effect=_fake_rank),
        patch("triage.pipeline.work", side_effect=_fake_work),
    ]
    for p in patches:
        p.start()
    try:
        _, run_id = pipe.run_pipeline(emails)
    finally:
        for p in patches:
            p.stop()

    items = from_pipeline.dump_run_to_agent_outputs(run_id)
    assert items[0]["router_output"]["category"] == "marketing"  # router's first take
    assert items[0]["evaluator_output"]["final_category"] == "risk"  # evaluator override


def test_runner_consumes_dump_end_to_end():
    """The output of from_pipeline must be consumable by evaluation/runner.py."""
    emails = [
        Message(id="e1", subject="x", sender="y"),
        Message(id="e2", subject="y", sender="z"),
        Message(id="e3", subject="z", sender="w"),
    ]
    _, run_id = _run_pipeline_with_fakes(emails)
    agent_items = from_pipeline.dump_run_to_agent_outputs(run_id)

    ground_truth = [
        {"id": "e1", "final_category": "work", "risk_score": 0.05, "needs_review": False},
        {"id": "e2", "final_category": "marketing", "risk_score": 0.02, "needs_review": False},
        {"id": "e3", "final_category": "risk", "risk_score": 0.95, "needs_review": True},
    ]
    cfg = EvaluatorConfig(
        high_risk_threshold=0.6,
        low_confidence_threshold=0.65,
        use_predicted_needs_review_field=True,
    )

    report = evaluate(ground_truth, agent_items, cfg=cfg)

    assert report["count"] == 3
    assert report["errors"] == []
    # all fakes match ground truth perfectly
    assert report["metrics"]["router"]["category_accuracy"] == 1.0
    assert report["metrics"]["evaluator"]["final_category_accuracy"] == 1.0
    assert report["metrics"]["evaluator"]["risk_score_mae"] == 0.0
    assert report["metrics"]["evaluator"]["needs_review_accuracy"] == 1.0


def test_dump_raises_when_no_runs():
    with pytest.raises(ValueError, match="no runs"):
        from_pipeline.dump_run_to_agent_outputs()
