from __future__ import annotations

import json
from pathlib import Path

from benchmarks.scoring.aggregate import aggregate, render_markdown


def _write_trial(path: Path, predictions: list[dict], calls: int = 5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scenario_id": path.parent.name,
        "agent": path.parent.parent.name,
        "per_email": predictions,
        "usage": {"calls": calls, "input_tokens": 100, "output_tokens": 50, "estimated_cost_usd": 0.0001},
    }))


_E1_PERFECT = [
    {"id": "e1_e01", "category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
    {"id": "e1_e02", "category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
    {"id": "e1_e03", "category": "marketing", "priority": "low", "action": "label", "needs_review": False},
    {"id": "e1_e04", "category": "work", "priority": "medium", "action": "flag", "needs_review": False},
    {"id": "e1_e05", "category": "risk", "priority": "high", "action": "flag", "needs_review": True},
]


def test_aggregate_perfect_run_table1_tsr_one(tmp_path):
    for trial in (1, 2, 3):
        _write_trial(tmp_path / "mailpilot" / "E1" / f"trial_{trial}.json", _E1_PERFECT)
    agg = aggregate(tmp_path)
    assert agg["table1"]["mailpilot"]["tsr"] == 1.0
    assert agg["table1"]["mailpilot"]["macro_f1"] > 0.99


def test_aggregate_easy_tier_in_table2(tmp_path):
    _write_trial(tmp_path / "rule" / "E1" / "trial_1.json", _E1_PERFECT)
    agg = aggregate(tmp_path)
    assert agg["table2"]["rule"]["easy"] == 1.0
    assert agg["table2"]["rule"]["medium"] == 0.0


def test_aggregate_asserts_trial_count(tmp_path):
    _write_trial(tmp_path / "rule" / "E1" / "trial_1.json", _E1_PERFECT)
    _write_trial(tmp_path / "rule" / "E1" / "trial_2.json", _E1_PERFECT)
    try:
        aggregate(tmp_path, expected_trials=3)
        assert False, "should have raised"
    except RuntimeError as exc:
        assert "expected 3" in str(exc)


def test_render_markdown_has_two_tables(tmp_path):
    _write_trial(tmp_path / "mailpilot" / "E1" / "trial_1.json", _E1_PERFECT)
    agg = aggregate(tmp_path)
    md = render_markdown(tmp_path, agg)
    assert "## Table 1" in md
    assert "## Table 2" in md
    assert "Easy" in md
