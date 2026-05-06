from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mailpilot.settings")

import django

django.setup()

from triage import persistence  # noqa: E402
from triage.agent import _usage  # noqa: E402
from triage.agent._client import default_model  # noqa: E402
from triage.pipeline import run_pipeline  # noqa: E402
from triage.schemas import Message  # noqa: E402


# Approximate published per-1M-token pricing in USD (cache-miss input, output).
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.14, 0.28),
    "claude-sonnet-4-6": (3.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, outp = _PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * inp + output_tokens * outp) / 1_000_000


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


SCEN_DIR = REPO_ROOT / "benchmarks" / "scenarios"
GT_DIR = REPO_ROOT / "benchmarks" / "ground_truth"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"



def load_scenario(scenario_id: str) -> dict:
    matches = list(SCEN_DIR.glob(f"{scenario_id}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No scenario file found for {scenario_id} in {SCEN_DIR}")
    return json.loads(matches[0].read_text())


def load_ground_truth(scenario_id: str) -> dict | None:
    matches = list(GT_DIR.glob(f"{scenario_id}_*.gt.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text())


def select_scenarios(difficulty: str, scenario: str | None) -> list[str]:
    if scenario:
        return [scenario.upper()]
    by_difficulty = {"easy": ["E1", "E2", "E3"],
                     "medium": ["M1", "M2", "M3", "M4"],
                     "hard": ["H1", "H2", "H3"]}
    if difficulty == "all":
        return by_difficulty["easy"] + by_difficulty["medium"] + by_difficulty["hard"]
    if difficulty not in by_difficulty:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return by_difficulty[difficulty]


def wipe_email_state(email_ids: list[str]) -> None:
    """Delete prior pipeline state for these email_ids so reruns are clean.

    Wipes three tables: email_state (per-email mutable fields), processed_emails
    (DONE marker that causes pipeline to skip), and email_snapshots (per-stage
    resume points that fast-forward past completed work).
    """
    if not email_ids:
        return
    conn = sqlite3.connect(persistence.DB_PATH)
    placeholders = ",".join("?" * len(email_ids))
    conn.execute(f"DELETE FROM email_state       WHERE email_id IN ({placeholders})", email_ids)
    conn.execute(f"DELETE FROM processed_emails  WHERE email_id IN ({placeholders})", email_ids)
    conn.execute(f"DELETE FROM email_snapshots   WHERE email_id IN ({placeholders})", email_ids)
    conn.commit()
    conn.close()


from benchmarks.baselines import mailpilot_full, no_evaluator, rule_based, single_prompt  # noqa: E402

AGENTS = {
    "mailpilot": mailpilot_full,
    "no_eval":   no_evaluator,
    "rule":      rule_based,
    "single":    single_prompt,
}


def run_one_scenario(
    scenario: dict,
    *,
    agent: str,
    max_emails: int | None,
    show_gt: bool,
) -> dict:
    sid = scenario["scenario_id"]
    name = scenario["scenario_name"]
    diff = scenario["difficulty"]
    emails = scenario["emails"]
    if max_emails:
        emails = emails[:max_emails]

    print(f"\n{'='*70}")
    print(f"=== {sid}: {name}  ({diff}, {len(emails)} email{'s' if len(emails)!=1 else ''}, agent={agent}) ===")
    print('='*70)

    # mailpilot + no_eval go through run_pipeline, which reads/writes sqlite;
    # wiping is harmless for the others.
    wipe_email_state([e["id"] for e in emails])

    msgs = [Message(**{**e, "thread_id": e.get("thread_id") or ""}) for e in emails]
    gt = load_ground_truth(sid) if show_gt else None
    gt_by_id = {row["id"]: row for row in (gt["per_email"] if gt else [])}

    _usage.reset()
    t0 = time.time()
    predictions = AGENTS[agent].run(msgs)
    elapsed = time.time() - t0
    usage = _usage.snapshot()
    model = default_model()
    usage["estimated_cost_usd"] = estimate_cost_usd(model, usage["input_tokens"], usage["output_tokens"])

    print(f"\ntime: {elapsed:.1f}s ({elapsed/len(emails):.1f}s/email)   "
          f"calls={usage['calls']} tokens={usage['input_tokens']}+{usage['output_tokens']} "
          f"≈${usage['estimated_cost_usd']:.4f}\n")

    correct = 0
    for i, p in enumerate(predictions, 1):
        subj = next((m.subject for m in msgs if m.id == p["id"]), "")[:65]
        print(f"[{i}/{len(msgs)}] {p['id']}  {subj}")
        print(f"    category={p['category']}  priority={p['priority']}  "
              f"action={p['action']}  needs_review={p['needs_review']}")
        gt_row = gt_by_id.get(p["id"])
        if gt_row:
            ok = (p["category"] == gt_row["category"]
                  and p["priority"] == gt_row["priority"]
                  and p["action"] == gt_row["action"]
                  and p["needs_review"] == gt_row["needs_review"])
            if ok:
                correct += 1
                print("    ✓ matches GT")
            else:
                diffs = [
                    f"{k}={p[k]}!=gt:{gt_row[k]}"
                    for k in ("category", "priority", "action", "needs_review")
                    if p[k] != gt_row[k]
                ]
                print(f"    ✗ {', '.join(diffs)}")
        p["ground_truth"] = gt_row
        print()

    if gt:
        print(f"GT score: {correct}/{len(msgs)} fully-correct rows "
              f"({100*correct/len(msgs):.0f}%)\n")

    return {
        "scenario_id": sid,
        "scenario_name": name,
        "difficulty": diff,
        "agent": agent,
        "elapsed_seconds": elapsed,
        "model": model,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": 0.0,
        "usage": usage,
        "per_email": predictions,
    }


def main():
    ap = argparse.ArgumentParser(description="Run MailPilot benchmark scenarios.")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="easy",
                    help="Run all scenarios at this difficulty (default: easy).")
    ap.add_argument("--scenario", default=None,
                    help="Run a single scenario by id (E1, M2, H3, ...). Overrides --difficulty.")
    ap.add_argument("--agent", choices=list(AGENTS), default="mailpilot",
                    help="Which agent to evaluate (default: mailpilot).")
    ap.add_argument("--trials", type=int, default=3,
                    help="How many trials per (agent, scenario). Paper uses 3-5.")
    ap.add_argument("--run-id", default=None,
                    help="Group output under results/<run-id>/. Reuse across --agent calls "
                         "to bundle multiple agents into one report.")
    ap.add_argument("--max-emails", type=int, default=None,
                    help="Cap emails per scenario for very fast smoke tests.")
    ap.add_argument("--compare", action="store_true",
                    help="Compare against ground truth and show ✓/✗ per email.")
    ap.add_argument("--max-cost-usd", type=float, default=None,
                    help="Abort the run if accumulated estimated cost crosses this cap.")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip writing report.md after the run.")
    args = ap.parse_args()

    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("ERROR: set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic).")
        sys.exit(1)
    print(f"Model: {default_model()}")

    persistence.init_db()

    sids = select_scenarios(args.difficulty, args.scenario)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"Agent: {args.agent}   trials: {args.trials}   scenarios: {', '.join(sids)}")

    cost_so_far = 0.0
    aborted = False
    for sid in sids:
        try:
            scen = load_scenario(sid)
        except FileNotFoundError as e:
            print(f"  skip {sid}: {e}")
            continue
        scen_dir = run_dir / args.agent / sid
        scen_dir.mkdir(parents=True, exist_ok=True)
        for trial in range(1, args.trials + 1):
            print(f"\n--- {args.agent}/{sid}/trial_{trial} ---")
            out = run_one_scenario(
                scen,
                agent=args.agent,
                max_emails=args.max_emails,
                show_gt=args.compare,
            )
            (scen_dir / f"trial_{trial}.json").write_text(
                json.dumps(out, indent=2, default=str)
            )
            cost_so_far += out["usage"]["estimated_cost_usd"]
            if args.max_cost_usd is not None and cost_so_far > args.max_cost_usd:
                print(f"\nABORT: estimated cost ${cost_so_far:.4f} exceeds cap ${args.max_cost_usd:.4f}")
                aborted = True
                break
        if aborted:
            break

    print(f"\nTotal estimated cost: ${cost_so_far:.4f}")

    if not args.no_aggregate and not aborted:
        from benchmarks.scoring.aggregate import aggregate, render_markdown
        try:
            agg = aggregate(run_dir)
            report = render_markdown(run_dir, agg)
            (run_dir / "report.md").write_text(report)
            print(f"\nReport: {run_dir / 'report.md'}\n")
            print(report)
        except RuntimeError as exc:
            print(f"\n(skipped report: {exc})")


if __name__ == "__main__":
    main()
