from __future__ import annotations

import json
import sys
from pathlib import Path

from .metrics import ACTIONS, CATEGORIES, PRIORITIES, REQUIRED_KEYS

GT_DIR = Path(__file__).resolve().parents[1] / "ground_truth"


def validate_one(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path.name}: not valid JSON ({exc})"]

    if "scenario_id" not in d:
        errors.append(f"{path.name}: missing scenario_id")
    if "per_email" not in d:
        errors.append(f"{path.name}: missing per_email")
        return errors

    for i, e in enumerate(d["per_email"]):
        prefix = f"{path.name}#{i} ({e.get('id', '?')})"
        for k in REQUIRED_KEYS:
            if k not in e:
                errors.append(f"{prefix}: missing key {k!r}")
        if e.get("category") not in CATEGORIES:
            errors.append(f"{prefix}: bad category {e.get('category')!r}")
        if e.get("priority") not in PRIORITIES:
            errors.append(f"{prefix}: bad priority {e.get('priority')!r}")
        if e.get("action") not in ACTIONS:
            errors.append(f"{prefix}: bad action {e.get('action')!r}")
        if not isinstance(e.get("needs_review"), bool):
            errors.append(f"{prefix}: needs_review must be bool")

    for i, t in enumerate(d.get("expected_tool_calls", [])):
        prefix = f"{path.name} tool#{i}"
        for k in ("id", "tool", "required_args"):
            if k not in t:
                errors.append(f"{prefix}: missing key {k!r}")
    return errors


def main() -> int:
    paths = sorted(GT_DIR.glob("*.gt.json"))
    if not paths:
        print(f"no GT files found in {GT_DIR}", file=sys.stderr)
        return 1
    all_errors = [e for p in paths for e in validate_one(p)]
    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        print(f"\n{len(all_errors)} validation error(s) across {len(paths)} GT file(s)", file=sys.stderr)
        return 1
    print(f"all {len(paths)} GT file(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
