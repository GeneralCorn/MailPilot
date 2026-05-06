from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triage import persistence
from triage.schemas import Category


def _latest_run_id() -> str | None:
    conn = persistence.get_conn()
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _emails_with_evaluator_snapshot(run_id: str) -> list[str]:
    conn = persistence.get_conn()
    rows = conn.execute(
        "SELECT DISTINCT email_id FROM email_snapshots "
        "WHERE run_id=? AND email_id IS NOT NULL AND stage IN ('evaluator', 'worker') "
        "ORDER BY email_id",
        (run_id,),
    ).fetchall()
    return [r[0] for r in rows]


def dump_run_to_agent_outputs(
    run_id: str | None = None, output_path: Path | None = None
) -> list[dict[str, Any]]:
    """Walk SQLite snapshots for a run, emit a list shaped for evaluation/runner.py.

    Each entry has id, router_output, evaluator_output. router_output is taken from the
    pre-evaluator router_outputs map (captured by the loop on first iteration); evaluator
    output is read from the latest snapshot (which may be a worker-stage snapshot — that's
    fine, the State only grows).
    """
    persistence.init_db(migrate_from_json=False)

    if run_id is None:
        run_id = _latest_run_id()
        if run_id is None:
            raise ValueError("no runs found in db; nothing to dump")

    out: list[dict[str, Any]] = []
    for email_id in _emails_with_evaluator_snapshot(run_id):
        last = persistence.latest_snapshot(run_id, email_id)
        if last is None:
            continue
        _, state = last

        router_output = state.router_outputs.get(email_id)
        if router_output is None:
            router_output = {
                "category": (
                    state.classifications.get(email_id, Category.UNCLASSIFIED).value
                    if hasattr(state.classifications.get(email_id, Category.UNCLASSIFIED), "value")
                    else str(state.classifications.get(email_id, "unclassified"))
                ),
                "confidence": state.confidence_scores.get(email_id, 0.0),
            }

        final_cat = state.classifications.get(email_id, Category.UNCLASSIFIED)
        evaluator_output = {
            "final_category": final_cat.value if hasattr(final_cat, "value") else str(final_cat),
            "confidence": state.confidence_scores.get(email_id, 0.0),
            "risk_score": state.risk_scores.get(email_id, 0.0),
            "needs_review": email_id in state.needs_review,
        }

        out.append({
            "id": email_id,
            "router_output": router_output,
            "evaluator_output": evaluator_output,
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump pipeline snapshots to evaluation/runner.py-compatible JSON."
    )
    parser.add_argument("--run-id", default=None, help="Run id; defaults to most recent run")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    items = dump_run_to_agent_outputs(args.run_id, Path(args.out))
    print(f"wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
