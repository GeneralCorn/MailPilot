from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.human_eval.irr import cohens_kappa, load_ratings
from benchmarks.human_eval.make_csv import emit_rows


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_cohens_kappa_random_agreement_near_zero():
    a = [1, 1, 1, 2, 2, 2]
    b = [2, 1, 1, 2, 1, 2]  # mixed
    k = cohens_kappa(a, b)
    assert -1.0 <= k <= 1.0


def test_cohens_kappa_systematic_disagreement_is_negative():
    # both raters use the same two values but always pick the opposite of each other
    assert cohens_kappa([1, 2, 1, 2], [2, 1, 2, 1]) < 0


def test_emit_rows_uses_4_criteria_per_email(tmp_path):
    trial = {
        "scenario_id": "E1",
        "agent": "rule",
        "per_email": [
            {"id": "x1", "category": "marketing", "priority": "low", "action": "archive", "needs_review": False},
            {"id": "x2", "category": "work", "priority": "high", "action": "flag", "needs_review": False},
        ],
        "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
    }
    p = tmp_path / "rule" / "E1" / "trial_1.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(trial))
    rows = emit_rows(tmp_path, "rule")
    assert len(rows) == 2 * 4  # 2 emails × 4 criteria


def test_load_ratings_skips_unrated_rows(tmp_path):
    csv_path = tmp_path / "r.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "email_id", "criterion", "rater_a", "rater_b", "notes"])
        w.writeheader()
        w.writerow({"scenario_id": "E1", "email_id": "x1", "criterion": "explanation_clarity", "rater_a": "5", "rater_b": "4", "notes": ""})
        w.writerow({"scenario_id": "E1", "email_id": "x1", "criterion": "explanation_clarity", "rater_a": "", "rater_b": "4", "notes": ""})
        w.writerow({"scenario_id": "E1", "email_id": "x1", "criterion": "signal_grounding", "rater_a": "3", "rater_b": "3", "notes": ""})

    grouped = load_ratings(csv_path)
    assert set(grouped) == {"explanation_clarity", "signal_grounding"}
    assert grouped["explanation_clarity"] == ([5], [4])
    assert grouped["signal_grounding"] == ([3], [3])
