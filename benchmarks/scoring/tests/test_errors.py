from __future__ import annotations

from benchmarks.scoring.errors import errors_by_type, ranking_inversions


_GT = [
    {"id": "a", "category": "marketing", "priority": "low",    "action": "archive",  "needs_review": False},
    {"id": "b", "category": "work",      "priority": "high",   "action": "flag",     "needs_review": False},
    {"id": "c", "category": "risk",      "priority": "high",   "action": "escalate", "needs_review": True},
    {"id": "d", "category": "billing",   "priority": "medium", "action": "flag",     "needs_review": False},
]


def _perfect():
    return [dict(g) for g in _GT]


def test_perfect_run_zero_errors():
    e = errors_by_type(_perfect(), _GT, expected_tool_calls=[])
    assert e["schema"]["count"] == 0
    assert e["category"]["count"] == 0
    assert e["ranking"]["count"] == 0
    assert e["risk_handling"]["count"] == 0


def test_category_error_counted():
    pred = _perfect()
    pred[0] = {**pred[0], "category": "personal"}
    e = errors_by_type(pred, _GT, [])
    assert e["category"]["count"] == 1
    assert e["category"]["rate"] == 1 / 4


def test_risk_handling_error_counted():
    pred = _perfect()
    pred[2] = {**pred[2], "action": "archive"}  # unsafe risk action
    e = errors_by_type(pred, _GT, [])
    assert e["risk_handling"]["count"] == 1
    assert e["risk_handling"]["total"] == 1


def test_ranking_inversion_counted():
    # swap b (gt high) with a (gt low) priorities → inversion
    pred = _perfect()
    pred[0] = {**pred[0], "priority": "high"}
    pred[1] = {**pred[1], "priority": "low"}
    inv, total = ranking_inversions(pred, _GT)
    assert inv > 0


def test_tool_action_miss_counted():
    pred = [{
        "id": "x",
        "category": "work", "priority": "high", "action": "calendar", "needs_review": False,
        "proposed_actions": [],
    }]
    gt = [{"id": "x", "category": "work", "priority": "high", "action": "calendar", "needs_review": False}]
    expected = [{"id": "x", "tool": "calendar.create_event", "required_args": ["title", "start", "end"]}]
    e = errors_by_type(pred, gt, expected)
    assert e["tool_action"]["count"] == 1
    assert e["tool_action"]["total"] == 1
    assert e["tool_action"]["rate"] == 1.0


def test_schema_error_counted():
    pred = [{"id": "x"}]  # missing every field
    gt = [{"id": "x", "category": "work", "priority": "high", "action": "label", "needs_review": False}]
    e = errors_by_type(pred, gt, [])
    assert e["schema"]["count"] > 0
