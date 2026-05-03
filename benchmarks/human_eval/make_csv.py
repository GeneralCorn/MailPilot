from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CRITERIA = ("explanation_clarity", "signal_grounding", "action_coherence", "batch_consistency")


def emit_rows(run_dir: Path, agent: str) -> list[dict]:
    agent_dir = run_dir / agent
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"no agent dir at {agent_dir}")
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for scen_dir in sorted(p for p in agent_dir.iterdir() if p.is_dir()):
        # one trial is enough for rating; pick the first.
        trials = sorted(scen_dir.glob("trial_*.json"))
        if not trials:
            continue
        trial = json.loads(trials[0].read_text())
        for e in trial["per_email"]:
            for c in CRITERIA:
                key = (e["id"], c)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "scenario_id": scen_dir.name,
                    "email_id": e["id"],
                    "criterion": c,
                    "rater_a": "",
                    "rater_b": "",
                    "notes": "",
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit a blank human-rating CSV from a benchmark run.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = emit_rows(args.run_dir, args.agent)
    out = args.out or (args.run_dir / f"{args.agent}_ratings.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "email_id", "criterion", "rater_a", "rater_b", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
