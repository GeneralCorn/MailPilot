from __future__ import annotations

import json
import re
from typing import Any, Callable


def _strip_code_fences(text: str) -> str:
    # Remove common Markdown code fences like ```json ... ```
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_json_object(value: Any) -> dict[str, Any]:
    """
    Convert an Agent output field into a JSON object.

    Accepts:
    - dict already
    - raw JSON string
    - string with code fences
    - string containing extra text; we take the first {...} block
    """
    if value is None:
        raise ValueError("value is None")
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Unsupported JSON value type: {type(value)}")

    text = _strip_code_fences(value)
    # Fast path: whole string is JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Parsed JSON is not an object")
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} substring.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    snippet = text[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON snippet is not an object")
    return parsed


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f"}:
            return False
    raise ValueError(f"Cannot convert to bool: {v!r}")


def to_float01(v: Any) -> float:
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        x = float(v.strip())
    else:
        raise ValueError(f"Cannot convert to float: {v!r}")
    # Clamp to [0, 1] to avoid metrics crashing on bad model outputs.
    return max(0.0, min(1.0, x))


def normalize_category(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"Category must be a string, got {type(v)}")
    s = v.strip().lower()
    # Accept common aliases.
    aliases = {
        "personal_email": "personal",
        "work_email": "work",
        "marketing_email": "marketing",
        "unclassifed": "unclassified",
        "unclassified": "unclassified",
    }
    return aliases.get(s, s)


def get_predicted_fields(
    agent_item: dict[str, Any],
    *,
    raw_field: str = "raw_output",
    output_field: str = "output",
) -> dict[str, Any]:
    """
    Normalize agent item to a single dict containing parsed evaluator JSON.

    Supported input shapes:
    - { "id": "...", "output": { ... } }
    - { "id": "...", "output": "<json string>" }
    - { "id": "...", "raw_output": "<json string>" }
    - { "id": "...", ...<evaluator json keys at top level>... }
    """
    if output_field in agent_item:
        return extract_json_object(agent_item[output_field])
    if raw_field in agent_item:
        return extract_json_object(agent_item[raw_field])

    # Fallback: assume top-level already looks like evaluator output JSON.
    # We only return it if it has at least one expected key.
    for k in ("final_category", "category", "risk_score", "confidence", "needs_review"):
        if k in agent_item:
            return dict(agent_item)
    raise KeyError(f"Cannot locate agent output JSON in keys: {raw_field!r}, {output_field!r}")


def get_gt_fields(gt_item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize ground truth item to expected keys for evaluator-level metrics.

    Expected GT keys (minimum):
    - final_category (or category)
    - risk_score
    - needs_review (optional; if missing we will compute it)
    """
    out: dict[str, Any] = {}
    if "final_category" in gt_item:
        out["final_category"] = gt_item["final_category"]
    elif "category" in gt_item:
        out["final_category"] = gt_item["category"]
    else:
        raise KeyError("GT missing 'final_category'/'category'")

    if "risk_score" in gt_item:
        out["risk_score"] = gt_item["risk_score"]
    else:
        raise KeyError("GT missing 'risk_score'")

    if "needs_review" in gt_item:
        out["needs_review"] = gt_item["needs_review"]
    return out


def compute_needs_review(
    *,
    final_category: str,
    risk_score: float,
    confidence: float | None = None,
    high_risk_threshold: float = 0.6,
    low_confidence_threshold: float = 0.65,
) -> bool:
    """
    Mirror the evaluator gating rule used in the repo's `triage/agent/evaluator.py`:
    - needs_review=true if:
      - risk_score > high_risk_threshold OR
      - (optional) confidence < low_confidence_threshold
    """
    if risk_score > high_risk_threshold:
        return True
    if confidence is not None and confidence < low_confidence_threshold:
        return True
    return False


def metric_category_accuracy(gt: dict[str, Any], pred: dict[str, Any]) -> bool:
    gt_cat = normalize_category(gt["final_category"])
    pred_cat = normalize_category(pred["final_category"])
    return gt_cat == pred_cat


def metric_risk_score_mae(gt: dict[str, Any], pred: dict[str, Any]) -> float:
    gt_r = to_float01(gt["risk_score"])
    pred_r = to_float01(pred["risk_score"])
    return abs(pred_r - gt_r)


def _confusion_counts(gt: bool, pred: bool) -> dict[str, int]:
    tp = 1 if (gt and pred) else 0
    fp = 1 if ((not gt) and pred) else 0
    fn = 1 if (gt and (not pred)) else 0
    tn = 1 if ((not gt) and (not pred)) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metric_needs_review_prf1(gt: bool, pred: bool) -> dict[str, float]:
    counts = _confusion_counts(gt, pred)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metric_needs_review_accuracy(gt: bool, pred: bool) -> float:
    return 1.0 if gt == pred else 0.0


def safe_get_number(d: dict[str, Any], key: str, default: float | None = None) -> float | None:
    if key not in d:
        return default
    return float(d[key])


def build_extractor(key: str, caster: Callable[[Any], Any]) -> Callable[[dict[str, Any]], Any]:
    def _extract(d: dict[str, Any]) -> Any:
        if key not in d:
            raise KeyError(f"Missing key: {key}")
        return caster(d[key])

    return _extract

