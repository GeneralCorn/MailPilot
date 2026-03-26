from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# When running as `python evaluation/runner.py`, Python puts `evaluation/` on sys.path,
# which breaks absolute imports like `import evaluation.*`. Ensure the repo root is present.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    compute_needs_review,
    get_gt_fields,
    get_predicted_fields,
    metric_needs_review_accuracy,
    normalize_category,
    to_float01,
    to_bool,
)


def _load_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("items", "data", "examples"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        # Common shape: { "ground_truth": [...] }
        for v in raw.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"Unsupported JSON format in {path}")


def _get_id(item: dict[str, Any]) -> str:
    for k in ("id", "email_id", "message_id"):
        if k in item:
            return str(item[k])
    raise KeyError("Item missing id/email_id/message_id")


def _pick_category_from_pred(pred_fields: dict[str, Any]) -> str:
    if "final_category" in pred_fields:
        return str(pred_fields["final_category"])
    if "category" in pred_fields:
        return str(pred_fields["category"])
    raise KeyError("Predicted output missing final_category/category")


def _pick_risk_score_from_pred(pred_fields: dict[str, Any]) -> float:
    if "risk_score" not in pred_fields:
        raise KeyError("Predicted output missing risk_score")
    return to_float01(pred_fields["risk_score"])


def _pick_confidence_from_pred(pred_fields: dict[str, Any]) -> float | None:
    if "confidence" not in pred_fields:
        return None
    return to_float01(pred_fields["confidence"])


def _pick_needs_review_from_pred(
    pred_fields: dict[str, Any],
    *,
    use_predicted_field: bool,
    high_risk_threshold: float,
    low_confidence_threshold: float,
) -> bool:
    if use_predicted_field and "needs_review" in pred_fields:
        return to_bool(pred_fields["needs_review"])

    final_category = normalize_category(_pick_category_from_pred(pred_fields))
    risk_score = _pick_risk_score_from_pred(pred_fields)
    confidence = _pick_confidence_from_pred(pred_fields)
    return compute_needs_review(
        final_category=final_category,
        risk_score=risk_score,
        confidence=confidence,
        high_risk_threshold=high_risk_threshold,
        low_confidence_threshold=low_confidence_threshold,
    )


@dataclass(frozen=True)
class EvaluatorConfig:
    high_risk_threshold: float
    low_confidence_threshold: float
    use_predicted_needs_review_field: bool


def _extract_stage_pred_fields(
    agent_item: dict[str, Any],
    *,
    stage: str,
    raw_field: str,
    output_field: str,
    router_raw_key: str,
    router_output_key: str,
    evaluator_raw_key: str,
    evaluator_output_key: str,
) -> dict[str, Any]:
    """
    Extract the predicted JSON object for a stage.

    Supports agent output shapes like:
    - { "router_raw_output": "<json string>", "evaluator_raw_output": "<json string>" }
    - { "router_output": { ... }, "evaluator_output": { ... } }
    - { "router": { "raw_output": "<json string>" }, "evaluator": { "raw_output": "<json string>" } }
    """
    if stage == "router":
        if router_raw_key in agent_item or router_output_key in agent_item:
            blob: dict[str, Any] = {}
            if router_output_key in agent_item:
                blob[output_field] = agent_item[router_output_key]
            if router_raw_key in agent_item:
                blob[raw_field] = agent_item[router_raw_key]
            # Prefer raw_output/output if present.
            if raw_field in blob:
                return get_predicted_fields(blob, raw_field=raw_field, output_field=output_field)
            return get_predicted_fields(blob, raw_field=raw_field, output_field=output_field)
    if stage == "evaluator":
        if evaluator_raw_key in agent_item or evaluator_output_key in agent_item:
            blob = {}
            if evaluator_output_key in agent_item:
                blob[output_field] = agent_item[evaluator_output_key]
            if evaluator_raw_key in agent_item:
                blob[raw_field] = agent_item[evaluator_raw_key]
            if raw_field in blob:
                return get_predicted_fields(blob, raw_field=raw_field, output_field=output_field)
            return get_predicted_fields(blob, raw_field=raw_field, output_field=output_field)

    # Nested shape: { "router": { ... }, "evaluator": { ... } }
    stage_blob = agent_item.get(stage)
    if isinstance(stage_blob, dict) and (raw_field in stage_blob or output_field in stage_blob):
        return get_predicted_fields(stage_blob, raw_field=raw_field, output_field=output_field)

    # Fallback: pass through and let get_predicted_fields attempt to parse.
    return get_predicted_fields(agent_item, raw_field=raw_field, output_field=output_field)


def _maybe_compute_gt_needs_review(gt_fields: dict[str, Any], cfg: EvaluatorConfig) -> bool:
    """
    Ground truth may omit needs_review/confidence.
    Align with evaluator gating semantics when needs_review is missing:
    - compute using risk_score threshold only
    """
    if "needs_review" in gt_fields:
        return to_bool(gt_fields["needs_review"])
    return compute_needs_review(
        final_category=normalize_category(gt_fields["final_category"]),
        risk_score=to_float01(gt_fields["risk_score"]),
        confidence=None,
        high_risk_threshold=cfg.high_risk_threshold,
        low_confidence_threshold=cfg.low_confidence_threshold,
    )


def evaluate(
    ground_truth_items: list[dict[str, Any]],
    agent_items: list[dict[str, Any]],
    *,
    cfg: EvaluatorConfig,
    router_raw_key: str = "router_raw_output",
    evaluator_raw_key: str = "evaluator_raw_output",
    router_output_key: str = "router_output",
    evaluator_output_key: str = "evaluator_output",
) -> dict[str, Any]:
    gt_by_id = {_get_id(it): it for it in ground_truth_items}
    pred_by_id = {_get_id(it): it for it in agent_items}

    ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    missing_in_pred = sorted(set(gt_by_id.keys()) - set(pred_by_id.keys()))
    missing_in_gt = sorted(set(pred_by_id.keys()) - set(gt_by_id.keys()))

    per_sample: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Router metrics
    router_cat_correct = 0
    router_conf_abs_err_sum = 0.0
    router_conf_abs_err_count = 0

    # Evaluator metrics
    tp = fp = fn = tn = 0  # needs_review confusion matrix
    evaluator_cat_correct = 0
    risk_abs_err_sum = 0.0
    risk_abs_err_count = 0

    for id_ in ids:
        gt_item = gt_by_id[id_]
        agent_item = pred_by_id[id_]
        try:
            gt_fields = get_gt_fields(gt_item)

            # ---- Router ----
            router_pred_fields = _extract_stage_pred_fields(
                agent_item,
                stage="router",
                raw_field="raw_output",
                output_field="output",
                router_raw_key=router_raw_key,
                router_output_key=router_output_key,
                evaluator_raw_key=evaluator_raw_key,
                evaluator_output_key=evaluator_output_key,
            )
            gt_final_category = normalize_category(gt_fields["final_category"])
            router_pred_category = normalize_category(_pick_category_from_pred(router_pred_fields))
            router_pred_confidence = _pick_confidence_from_pred(router_pred_fields)

            is_router_cat_correct = gt_final_category == router_pred_category
            if is_router_cat_correct:
                router_cat_correct += 1

            # if GT later adds router confidence, we can compute it; keep optional
            if "confidence" in gt_fields and router_pred_confidence is not None:
                router_conf_abs_err_sum += abs(float(gt_fields["confidence"]) - router_pred_confidence)
                router_conf_abs_err_count += 1

            # ---- Evaluator ----
            evaluator_pred_fields = _extract_stage_pred_fields(
                agent_item,
                stage="evaluator",
                raw_field="raw_output",
                output_field="output",
                router_raw_key=router_raw_key,
                router_output_key=router_output_key,
                evaluator_raw_key=evaluator_raw_key,
                evaluator_output_key=evaluator_output_key,
            )

            evaluator_pred_final_category = normalize_category(_pick_category_from_pred(evaluator_pred_fields))
            evaluator_pred_risk_score = _pick_risk_score_from_pred(evaluator_pred_fields)
            evaluator_pred_confidence = _pick_confidence_from_pred(evaluator_pred_fields)

            evaluator_pred_needs_review = _pick_needs_review_from_pred(
                evaluator_pred_fields,
                use_predicted_field=cfg.use_predicted_needs_review_field,
                high_risk_threshold=cfg.high_risk_threshold,
                low_confidence_threshold=cfg.low_confidence_threshold,
            )

            gt_needs_review = _maybe_compute_gt_needs_review(gt_fields, cfg)

            is_evaluator_cat_correct = gt_final_category == evaluator_pred_final_category
            if is_evaluator_cat_correct:
                evaluator_cat_correct += 1

            gt_risk_score = to_float01(gt_fields["risk_score"])
            risk_abs_err_sum += abs(evaluator_pred_risk_score - gt_risk_score)
            risk_abs_err_count += 1

            if gt_needs_review and evaluator_pred_needs_review:
                tp += 1
            elif (not gt_needs_review) and evaluator_pred_needs_review:
                fp += 1
            elif gt_needs_review and (not evaluator_pred_needs_review):
                fn += 1
            else:
                tn += 1

            per_sample.append(
                {
                    "id": id_,
                    "gt": gt_fields,
                    "pred": {
                        "router": {
                            "category": router_pred_category,
                            "confidence": router_pred_confidence,
                        },
                        "evaluator": {
                            "final_category": evaluator_pred_final_category,
                            "confidence": evaluator_pred_confidence,
                            "risk_score": evaluator_pred_risk_score,
                            "needs_review": evaluator_pred_needs_review,
                        },
                    },
                    "metrics": {
                        "router_category_correct": is_router_cat_correct,
                        "evaluator_category_correct": is_evaluator_cat_correct,
                        "evaluator_risk_score_abs_error": abs(evaluator_pred_risk_score - gt_risk_score),
                        "needs_review_correct": metric_needs_review_accuracy(gt_needs_review, evaluator_pred_needs_review),
                    },
                }
            )
        except Exception as e:
            errors.append({"id": id_, "error": str(e)})

    router_category_accuracy = router_cat_correct / len(ids) if ids else 0.0
    router_conf_mae = router_conf_abs_err_sum / router_conf_abs_err_count if router_conf_abs_err_count else None
    evaluator_category_accuracy = evaluator_cat_correct / len(ids) if ids else 0.0
    risk_score_mae = risk_abs_err_sum / risk_abs_err_count if risk_abs_err_count else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    needs_review_accuracy = (tp + tn) / len(ids) if ids else 0.0

    return {
        "count": len(ids),
        "missing_in_pred": missing_in_pred,
        "missing_in_gt": missing_in_gt,
        "errors": errors,
        "metrics": {
            "router": {
                "category_accuracy": router_category_accuracy,
                "confidence_mae": router_conf_mae,
            },
            "evaluator": {
                "final_category_accuracy": evaluator_category_accuracy,
                "risk_score_mae": risk_score_mae,
                "needs_review_precision": precision,
                "needs_review_recall": recall,
                "needs_review_f1": f1,
                "needs_review_accuracy": needs_review_accuracy,
            },
        },
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent outputs vs Ground Truth JSON.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("--agent-outputs", required=True, help="Path to agent_outputs.json")
    parser.add_argument("--report", required=False, default="", help="Optional output report JSON path")
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=0.6,
        help="Risk score threshold for needs_review gating",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for needs_review gating",
    )
    parser.add_argument(
        "--needs-review-source",
        choices=["field", "computed"],
        default="field",
        help="Use predicted needs_review field from agent output, or compute from other fields.",
    )
    parser.add_argument("--router-raw-key", default="router_raw_output", help="Key for router raw output JSON string")
    parser.add_argument("--router-output-key", default="router_output", help="Key for router structured output dict")
    parser.add_argument("--evaluator-raw-key", default="evaluator_raw_output", help="Key for evaluator raw output JSON string")
    parser.add_argument("--evaluator-output-key", default="evaluator_output", help="Key for evaluator structured output dict")

    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    agent_path = Path(args.agent_outputs)

    gt_items = _load_items(gt_path)
    agent_items = _load_items(agent_path)

    cfg = EvaluatorConfig(
        high_risk_threshold=args.high_risk_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        use_predicted_needs_review_field=(args.needs_review_source == "field"),
    )

    report = evaluate(
        gt_items,
        agent_items,
        cfg=cfg,
        router_raw_key=args.router_raw_key,
        router_output_key=args.router_output_key,
        evaluator_raw_key=args.evaluator_raw_key,
        evaluator_output_key=args.evaluator_output_key,
    )

    if args.report:
        out_path = Path(args.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

