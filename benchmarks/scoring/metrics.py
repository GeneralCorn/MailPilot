from __future__ import annotations

from collections import Counter

CATEGORIES = ("marketing", "personal", "work", "risk", "billing")
PRIORITIES = ("high", "medium", "low")
ACTIONS = ("label", "flag", "archive", "reply_draft", "calendar", "escalate", "no_action")
REQUIRED_KEYS = ("id", "category", "priority", "action", "needs_review")


def validate_schema(predicted: list[dict]) -> list[str]:
    errors: list[str] = []
    for i, e in enumerate(predicted):
        eid = e.get("id", "?")
        for k in REQUIRED_KEYS:
            if k not in e:
                errors.append(f"row {i} ({eid}): missing key '{k}'")
        if e.get("category") not in CATEGORIES:
            errors.append(f"row {i} ({eid}): bad category {e.get('category')!r}")
        if e.get("priority") not in PRIORITIES:
            errors.append(f"row {i} ({eid}): bad priority {e.get('priority')!r}")
        if e.get("action") not in ACTIONS:
            errors.append(f"row {i} ({eid}): bad action {e.get('action')!r}")
        if not isinstance(e.get("needs_review"), bool):
            errors.append(f"row {i} ({eid}): needs_review must be bool")
    return errors


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def _aligned_pairs(predicted: list[dict], gt: list[dict], field: str) -> list[tuple[str, str]]:
    pred_by_id = _by_id(predicted)
    return [(g.get(field, ""), pred_by_id.get(g["id"], {}).get(field, "")) for g in gt]


def confusion_matrix(predicted: list[dict], gt: list[dict], labels: tuple[str, ...]) -> dict:
    cm: Counter = Counter()
    for true_v, pred_v in _aligned_pairs(predicted, gt, "category"):
        cm[(true_v, pred_v)] += 1
    return {(t, p): cm[(t, p)] for t in labels for p in labels}


def macro_f1(predicted: list[dict], gt: list[dict], labels: tuple[str, ...] = CATEGORIES) -> float:
    pairs = _aligned_pairs(predicted, gt, "category")
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        if tp + fp + fn == 0:
            continue
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def tier_accuracy(predicted: list[dict], gt: list[dict]) -> float:
    pairs = _aligned_pairs(predicted, gt, "priority")
    if not pairs:
        return 0.0
    return sum(1 for t, p in pairs if t == p) / len(pairs)


def action_accuracy(predicted: list[dict], gt: list[dict]) -> float:
    pairs = _aligned_pairs(predicted, gt, "action")
    if not pairs:
        return 0.0
    return sum(1 for t, p in pairs if t == p) / len(pairs)
