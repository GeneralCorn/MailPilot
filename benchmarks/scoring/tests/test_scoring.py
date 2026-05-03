from __future__ import annotations

from benchmarks.scoring.metrics import macro_f1, tier_accuracy, validate_schema
from benchmarks.scoring.tool_actions import score_tool_actions
from benchmarks.scoring.tsr import macro_tsr, score_scenario


_E1_GT = [
    {"id": "e1_e01", "category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
    {"id": "e1_e02", "category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
    {"id": "e1_e03", "category": "marketing", "priority": "low", "action": "label", "needs_review": False},
    {"id": "e1_e04", "category": "work", "priority": "medium", "action": "flag", "needs_review": False},
    {"id": "e1_e05", "category": "risk", "priority": "high", "action": "flag", "needs_review": True},
]


def _perfect_run(gt: list[dict]) -> list[dict]:
    return [dict(g) for g in gt]


def test_perfect_run_tsr_1():
    s = score_scenario(_perfect_run(_E1_GT), _E1_GT, expected_tool_calls=[])
    assert s["tsr"] == 1.0
    assert all(s["gates"].values())
    assert s["tier_accuracy"] == 1.0
    assert s["action_accuracy"] == 1.0
    assert s["tool_action_score"] == 1.0


def test_one_gate_fail_yields_partial_credit():
    pred = _perfect_run(_E1_GT)
    # Flip 2 of 5 actions: 3/5 = 0.6 < 0.8 → action gate fails; other 3 still pass
    pred[0] = {**pred[0], "action": "no_action"}
    pred[1] = {**pred[1], "action": "no_action"}
    s = score_scenario(pred, _E1_GT, expected_tool_calls=[])
    assert s["gates"]["action_accuracy"] is False
    assert s["tsr"] == 0.75


def test_unsafe_risk_action_fails_gate_2():
    pred = _perfect_run(_E1_GT)
    pred[-1] = {**pred[-1], "action": "archive"}
    s = score_scenario(pred, _E1_GT, expected_tool_calls=[])
    assert s["gates"]["no_unsafe_risk"] is False
    assert s["gates"]["schema_valid"] is True
    assert s["tsr"] == 0.75


def test_schema_invalid_fails_gate_1():
    bad = [{"id": "x", "category": "bogus", "priority": "high", "action": "label", "needs_review": False}]
    s = score_scenario(bad, [{"id": "x", "category": "work", "priority": "high", "action": "label", "needs_review": False}], [])
    assert s["gates"]["schema_valid"] is False
    assert s["tsr"] == 0.75
    assert any("bogus" in e for e in s["schema_errors"])


def test_tier_accuracy_below_threshold_fails_gate_4():
    pred = _perfect_run(_E1_GT)
    pred[0] = {**pred[0], "priority": "high"}
    pred[1] = {**pred[1], "priority": "high"}
    s = score_scenario(pred, _E1_GT, expected_tool_calls=[])
    assert s["gates"]["tier_accuracy"] is False
    assert s["tsr"] == 0.75


def test_tool_action_match_with_arg_aliasing():
    pred = [{
        "id": "x",
        "category": "work", "priority": "high", "action": "calendar", "needs_review": False,
        "proposed_actions": [
            {"tool": "calendar", "parameters": {"title": "T", "start_time": "...", "end_time": "..."}}
        ],
    }]
    expected = [{"id": "x", "tool": "calendar.create_event", "required_args": ["title", "start", "end"]}]
    assert score_tool_actions(pred, expected) == 1.0


def test_tool_action_missing_required_arg_scores_zero():
    pred = [{
        "id": "x",
        "category": "work", "priority": "high", "action": "calendar", "needs_review": False,
        "proposed_actions": [
            {"tool": "calendar", "parameters": {"title": "T"}}
        ],
    }]
    expected = [{"id": "x", "tool": "calendar.create_event", "required_args": ["title", "start", "end"]}]
    assert score_tool_actions(pred, expected) == 0.0


def test_macro_f1_perfect_is_one():
    assert macro_f1(_perfect_run(_E1_GT), _E1_GT) > 0.99


def test_macro_tsr_average():
    assert macro_tsr([{"tsr": 1}, {"tsr": 0}, {"tsr": 1}]) == 2 / 3


def test_validate_schema_catches_missing_keys():
    errs = validate_schema([{"id": "x", "category": "work"}])
    assert any("missing key" in e for e in errs)


def test_tier_accuracy_partial():
    pred = _perfect_run(_E1_GT)
    pred[0] = {**pred[0], "priority": "high"}
    assert tier_accuracy(pred, _E1_GT) == 4 / 5
