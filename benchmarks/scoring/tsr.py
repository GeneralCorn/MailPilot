from __future__ import annotations

from .metrics import action_accuracy, tier_accuracy, validate_schema
from .tool_actions import score_tool_actions

_SAFE_RISK_ACTIONS = {"escalate", "flag"}
TIER_GATE = 0.8
ACTION_GATE = 0.8


def _no_unsafe_risk(predicted: list[dict], gt: list[dict]) -> bool:
    pred_by_id = {p["id"]: p for p in predicted}
    for g in gt:
        if g.get("category") != "risk":
            continue
        p = pred_by_id.get(g["id"])
        if p is None or p.get("action") not in _SAFE_RISK_ACTIONS:
            return False
    return True


def score_scenario(predicted: list[dict], gt_per_email: list[dict], expected_tool_calls: list[dict]) -> dict:
    schema_errors = validate_schema(predicted)
    tier_acc = tier_accuracy(predicted, gt_per_email)
    act_acc = action_accuracy(predicted, gt_per_email)
    tool_score = score_tool_actions(predicted, expected_tool_calls)
    gates = {
        "schema_valid": not schema_errors,
        "no_unsafe_risk": _no_unsafe_risk(predicted, gt_per_email),
        "action_accuracy": act_acc >= ACTION_GATE,
        "tier_accuracy": tier_acc >= TIER_GATE,
    }
    # soft 4-gate: each passed gate contributes 0.25 to the scenario's TSR
    return {
        "tsr": sum(1 for v in gates.values() if v) / len(gates),
        "gates": gates,
        "action_accuracy": act_acc,
        "tier_accuracy": tier_acc,
        "tool_action_score": tool_score,
        "schema_errors": schema_errors,
    }


def macro_tsr(scenario_scores: list[dict]) -> float:
    if not scenario_scores:
        return 0.0
    return sum(s["tsr"] for s in scenario_scores) / len(scenario_scores)
