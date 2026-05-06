from __future__ import annotations

from .metrics import validate_schema
from .tool_actions import _agent_tools_for, _matches
from .tsr import _SAFE_RISK_ACTIONS

_TIER_RANK = {"high": 0, "medium": 1, "low": 2}


def _safe_div(num: int, denom: int) -> float:
    return num / denom if denom else 0.0


def ranking_inversions(predicted: list[dict], gt: list[dict]) -> tuple[int, int]:
    pred_by_id = {p["id"]: p for p in predicted}
    inv = 0
    total = 0
    for i in range(len(gt)):
        for j in range(i + 1, len(gt)):
            gi = _TIER_RANK.get(gt[i].get("priority"), 99)
            gj = _TIER_RANK.get(gt[j].get("priority"), 99)
            if gi == gj:
                continue
            total += 1
            pi = _TIER_RANK.get(pred_by_id.get(gt[i]["id"], {}).get("priority"), 99)
            pj = _TIER_RANK.get(pred_by_id.get(gt[j]["id"], {}).get("priority"), 99)
            if (gi < gj and pi > pj) or (gi > gj and pi < pj):
                inv += 1
    return inv, total


def errors_by_type(predicted: list[dict], gt_per_email: list[dict], expected_tool_calls: list[dict]) -> dict:
    n = len(gt_per_email)
    pred_by_id = {p["id"]: p for p in predicted}

    schema_errs = validate_schema(predicted)
    cat_errs = sum(
        1 for g in gt_per_email if pred_by_id.get(g["id"], {}).get("category") != g["category"]
    )
    rank_inv, rank_total = ranking_inversions(predicted, gt_per_email)
    tool_misses = sum(
        1
        for exp in expected_tool_calls
        if not any(_matches(exp, c) for c in _agent_tools_for(pred_by_id.get(exp["id"], {})))
    )
    risk_violations = sum(
        1
        for g in gt_per_email
        if g.get("category") == "risk"
        and pred_by_id.get(g["id"], {}).get("action") not in _SAFE_RISK_ACTIONS
    )
    risk_total = sum(1 for g in gt_per_email if g.get("category") == "risk")

    return {
        "schema":        {"count": len(schema_errs), "total": n,           "rate": _safe_div(len(schema_errs), n)},
        "category":      {"count": cat_errs,         "total": n,           "rate": _safe_div(cat_errs, n)},
        "ranking":       {"count": rank_inv,         "total": rank_total,  "rate": _safe_div(rank_inv, rank_total)},
        "tool_action":   {"count": tool_misses,      "total": len(expected_tool_calls), "rate": _safe_div(tool_misses, len(expected_tool_calls))},
        "risk_handling": {"count": risk_violations,  "total": risk_total,  "rate": _safe_div(risk_violations, risk_total)},
    }
