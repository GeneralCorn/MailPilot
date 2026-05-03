"""Run benchmark scenarios through the live MailPilot pipeline and dump artifacts.

Quick testing usage:
    python3 benchmarks/run_benchmark.py --difficulty easy
    python3 benchmarks/run_benchmark.py --scenario E1
    python3 benchmarks/run_benchmark.py --difficulty all --compare
    python3 benchmarks/run_benchmark.py --scenario M3 --max-emails 3

What it does:
    1. Loads benchmark scenario JSON(s) from benchmarks/scenarios/
    2. Wipes prior pipeline state for those email IDs (so reruns are clean)
    3. Calls triage.pipeline.run_pipeline on each scenario's emails
    4. Reads per-email artifacts back from sqlite (and the live State)
    5. Prints a compact human-readable report and writes JSON results

Calendar / send_email / reply_draft become PROPOSALS — they do not fire
real Google API calls until you approve them in the UI. So this script is
safe to run repeatedly without polluting your inbox or calendar.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mailpilot.settings")

# Process --backend BEFORE django.setup() so settings.py sees the right env var
# and applies the ANTHROPIC_API_KEY stub when running on Tinker.
def _early_backend_override():
    for i, a in enumerate(sys.argv):
        if a == "--backend" and i + 1 < len(sys.argv):
            os.environ["MAILPILOT_LLM_BACKEND"] = sys.argv[i + 1]
            return
        if a.startswith("--backend="):
            os.environ["MAILPILOT_LLM_BACKEND"] = a.split("=", 1)[1]
            return
_early_backend_override()

import django  # noqa: E402

django.setup()

from triage import persistence  # noqa: E402
from triage.pipeline import run_pipeline  # noqa: E402
from triage.schemas import Message  # noqa: E402


SCEN_DIR = REPO_ROOT / "benchmarks" / "scenarios"
GT_DIR = REPO_ROOT / "benchmarks" / "ground_truth"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


# ── Loading ─────────────────────────────────────────────────────────────────

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


# ── State management ────────────────────────────────────────────────────────

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


# ── Action derivation ───────────────────────────────────────────────────────

def derive_primary_action(state_row: dict, in_needs_review: bool) -> str:
    """Reduce the agent's multi-faceted output to a single comparable 'action' label."""
    escalations = state_row.get("escalations") or []
    proposed = state_row.get("proposed_actions") or []
    external = state_row.get("external_actions") or []
    flagged = bool(state_row.get("flagged", 0))
    draft_reply = state_row.get("draft_reply")
    calendar = state_row.get("calendar_event")

    # 1. Risk gating wins above all (matches GT for needs_review=true rows)
    if escalations or in_needs_review:
        return "escalate"

    # 2. Proposed actions reflect what the worker decided but hasn't fired yet
    proposed_tools = [a.get("tool") for a in proposed if isinstance(a, dict)]
    if "calendar" in proposed_tools:
        return "calendar"
    if "send_email" in proposed_tools or "reply_draft" in proposed_tools or draft_reply:
        return "reply_draft"

    # 3. Auto-executed safe actions
    executed_tools = [a.get("tool") for a in external if isinstance(a, dict)]
    if "archive" in executed_tools:
        return "archive"
    if "label" in executed_tools:
        return "label"

    # 4. Flagging without escalation
    if flagged:
        return "flag"

    # 5. Calendar event already created (rare in benchmark — proposals usually pending)
    if calendar:
        return "calendar"

    return "no_action"


# ── Main run loop ───────────────────────────────────────────────────────────

def run_one_scenario(
    scenario: dict,
    *,
    max_emails: int | None,
    show_gt: bool,
    verbose: bool,
) -> dict:
    sid = scenario["scenario_id"]
    name = scenario["scenario_name"]
    diff = scenario["difficulty"]
    emails = scenario["emails"]
    if max_emails:
        emails = emails[:max_emails]

    print(f"\n{'='*70}")
    print(f"=== {sid}: {name}  ({diff}, {len(emails)} email{'s' if len(emails)!=1 else ''}) ===")
    print('='*70)

    wipe_email_state([e["id"] for e in emails])

    msgs = []
    for e in emails:
        # Message schema expects str for thread_id, but benchmarks use null for standalone emails
        normalized = {**e, "thread_id": e.get("thread_id") or ""}
        msgs.append(Message(**normalized))

    gt = load_ground_truth(sid) if show_gt else None
    gt_by_id = {row["id"]: row for row in (gt["per_email"] if gt else [])}

    t0 = time.time()
    state, run_id = run_pipeline(msgs)
    elapsed = time.time() - t0

    print(f"\n[run_id: {run_id}]   total time: {elapsed:.1f}s   ({elapsed/len(emails):.1f}s/email)\n")

    results: list[dict] = []
    correct_count = 0

    for i, msg in enumerate(msgs, 1):
        eid = msg.id
        row = persistence.get_email_state(eid)
        in_needs_review = eid in state.needs_review
        action = derive_primary_action(row, in_needs_review)

        category = row.get("category") or "?"
        priority = row.get("priority") or "?"
        status = row.get("status") or "?"
        flagged = bool(row.get("flagged", 0))
        escalations = row.get("escalations") or []
        proposed = row.get("proposed_actions") or []
        external = row.get("external_actions") or []
        summary = (row.get("summary") or "").strip()
        draft = (row.get("draft_reply") or "").strip()
        cal = row.get("calendar_event")

        # ── compact per-email console output ────────────────────────────────
        subj = (msg.subject or "")[:65]
        print(f"[{i}/{len(msgs)}] {eid}  {subj}")
        print(f"    category={category}  priority={priority}  status={status}")
        flag_label = "YES" if flagged else "no"
        review_label = "YES" if in_needs_review else "no"
        print(f"    primary_action={action}  flagged={flag_label}  needs_review={review_label}")

        if escalations:
            for esc in escalations:
                reason = esc.get("reason", "(no reason)") if isinstance(esc, dict) else str(esc)
                print(f"    !! escalation: {reason}")

        if proposed:
            for p in proposed:
                if not isinstance(p, dict): continue
                tool = p.get("tool", "?")
                params = p.get("parameters", {}) or {}
                # show only the most informative fields
                preview_keys = ["title", "to", "subject", "start_time", "end_time"]
                preview = {k: params[k] for k in preview_keys if k in params}
                print(f"    -> proposed: {tool}  {preview if preview else ''}")

        if external:
            for x in external:
                if not isinstance(x, dict): continue
                tool = x.get("tool", "?")
                ok = "ok" if x.get("success") else "failed"
                print(f"    -- executed: {tool} ({ok})")

        if summary and verbose:
            print(f"    summary: {summary[:120]}{'…' if len(summary)>120 else ''}")
        if draft and verbose:
            print(f"    draft: {draft[:120]}{'…' if len(draft)>120 else ''}")
        if cal and verbose:
            print(f"    calendar_event: {cal}")

        # ── ground truth comparison (optional) ──────────────────────────────
        gt_row = gt_by_id.get(eid)
        if gt_row:
            ok = (
                category == gt_row["category"]
                and priority == gt_row["priority"]
                and action == gt_row["action"]
                and in_needs_review == gt_row["needs_review"]
            )
            mark = "✓" if ok else "✗"
            if not ok:
                diffs = []
                if category != gt_row["category"]:
                    diffs.append(f"cat={category}!=gt:{gt_row['category']}")
                if priority != gt_row["priority"]:
                    diffs.append(f"pri={priority}!=gt:{gt_row['priority']}")
                if action != gt_row["action"]:
                    diffs.append(f"act={action}!=gt:{gt_row['action']}")
                if in_needs_review != gt_row["needs_review"]:
                    diffs.append(f"review={in_needs_review}!=gt:{gt_row['needs_review']}")
                print(f"    {mark} GT mismatch: {', '.join(diffs)}")
            else:
                correct_count += 1
                print(f"    {mark} matches GT")
        print()

        results.append({
            "id": eid,
            "subject": msg.subject,
            "category": category,
            "priority": priority,
            "status": status,
            "primary_action": action,
            "flagged": flagged,
            "needs_review": in_needs_review,
            "escalations": escalations,
            "proposed_actions": proposed,
            "external_actions": external,
            "summary": summary,
            "draft_reply": draft,
            "calendar_event": cal,
            "ground_truth": gt_row,
        })

    if gt:
        print(f"GT score: {correct_count}/{len(msgs)} fully-correct rows "
              f"({100*correct_count/len(msgs):.0f}%)\n")

    return {
        "scenario_id": sid,
        "scenario_name": name,
        "difficulty": diff,
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "per_email": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Run MailPilot benchmark scenarios.")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="easy",
                    help="Run all scenarios at this difficulty (default: easy).")
    ap.add_argument("--scenario", default=None,
                    help="Run a single scenario by id (E1, M2, H3, ...). Overrides --difficulty.")
    ap.add_argument("--max-emails", type=int, default=None,
                    help="Cap emails per scenario for very fast smoke tests.")
    ap.add_argument("--compare", action="store_true",
                    help="Compare against ground truth and show ✓/✗ per email.")
    ap.add_argument("--verbose", action="store_true",
                    help="Show summaries, drafts, and other long fields.")
    ap.add_argument("--output", default=None,
                    help="Path to write the JSON results (default: benchmarks/results/<timestamp>.json).")
    ap.add_argument("--backend", choices=["anthropic", "tinker"], default=None,
                    help="LLM backend. Defaults to MAILPILOT_LLM_BACKEND from .env, or 'anthropic'.")
    args = ap.parse_args()

    backend = (args.backend
               or os.environ.get("MAILPILOT_LLM_BACKEND")
               or "anthropic")
    if backend == "tinker":
        if not os.environ.get("TINKER_API_KEY"):
            print("ERROR: TINKER_API_KEY is not set. Add it to .env and rerun.")
            sys.exit(1)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "").startswith("tinker-stub"):
            print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env or pass --backend tinker.")
            sys.exit(1)
    print(f"LLM backend: {backend}")

    persistence.init_db()

    sids = select_scenarios(args.difficulty, args.scenario)
    print(f"Running {len(sids)} scenario(s): {', '.join(sids)}")
    print(f"Compare to ground truth: {args.compare}")

    all_results = []
    overall_correct = 0
    overall_total = 0
    for sid in sids:
        try:
            scen = load_scenario(sid)
        except FileNotFoundError as e:
            print(f"  skip {sid}: {e}")
            continue
        out = run_one_scenario(
            scen,
            max_emails=args.max_emails,
            show_gt=args.compare,
            verbose=args.verbose,
        )
        all_results.append(out)
        if args.compare:
            for er in out["per_email"]:
                gt = er.get("ground_truth")
                if not gt:
                    continue
                overall_total += 1
                if (er["category"] == gt["category"]
                        and er["priority"] == gt["priority"]
                        and er["primary_action"] == gt["action"]
                        and er["needs_review"] == gt["needs_review"]):
                    overall_correct += 1

    if args.compare and overall_total:
        print(f"\n{'='*70}")
        print(f"OVERALL: {overall_correct}/{overall_total} rows fully correct "
              f"({100*overall_correct/overall_total:.1f}%)")
        print('='*70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"run_{int(time.time())}.json"
    )
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()
