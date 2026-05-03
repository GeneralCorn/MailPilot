from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from .errors import errors_by_type
from .metrics import macro_f1, tier_accuracy
from .tsr import score_scenario

GT_DIR = Path(__file__).resolve().parents[1] / "ground_truth"

_TIER_OF_SCENARIO = {
    "E1": "easy", "E2": "easy", "E3": "easy",
    "M1": "medium", "M2": "medium", "M3": "medium", "M4": "medium",
    "H1": "hard", "H2": "hard", "H3": "hard",
}

_AGENT_LABELS = {
    "rule":      "Rule-Based Filter",
    "single":    "Single-Prompt LLM",
    "no_eval":   "MailPilot (No Evaluator)",
    "mailpilot": "MailPilot (full)",
}


def _load_gt(scenario_id: str) -> dict:
    matches = list(GT_DIR.glob(f"{scenario_id}_*.gt.json"))
    if not matches:
        raise FileNotFoundError(f"no GT for {scenario_id}")
    return json.loads(matches[0].read_text())


def load_trials(run_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """{(agent, scenario_id): [trial_dict, ...]}."""
    out: dict[tuple[str, str], list[dict]] = {}
    for agent_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for scen_dir in sorted(p for p in agent_dir.iterdir() if p.is_dir()):
            trials = sorted(scen_dir.glob("trial_*.json"))
            out[(agent_dir.name, scen_dir.name)] = [json.loads(t.read_text()) for t in trials]
    return out


def score_trial(trial: dict, gt: dict) -> dict:
    pred = trial["per_email"]
    expected = gt.get("expected_tool_calls", [])
    tsr = score_scenario(pred, gt["per_email"], expected)
    errs = errors_by_type(pred, gt["per_email"], expected)
    n = len(gt["per_email"])
    calls_per_email = trial["usage"]["calls"] / n if n else 0.0
    return {
        "tsr": tsr["tsr"],
        "macro_f1": macro_f1(pred, gt["per_email"]),
        "tier_accuracy": tier_accuracy(pred, gt["per_email"]),
        "risk_errors": errs["risk_handling"]["count"],
        "calls_per_email": calls_per_email,
    }


def _mean(vs: list[float]) -> float:
    return statistics.mean(vs) if vs else 0.0


def _stdev(vs: list[float]) -> float:
    return statistics.stdev(vs) if len(vs) > 1 else 0.0


def aggregate(run_dir: Path, expected_trials: int | None = None) -> dict:
    trials_by_cell = load_trials(run_dir)
    if not trials_by_cell:
        raise RuntimeError(f"no trial files found under {run_dir}")

    # per (agent, scenario) → list of per-trial scores
    per_cell: dict[tuple[str, str], list[dict]] = {}
    for (agent, scen), trials in trials_by_cell.items():
        if expected_trials is not None and len(trials) != expected_trials:
            raise RuntimeError(f"{agent}/{scen}: expected {expected_trials} trials, found {len(trials)}")
        gt = _load_gt(scen)
        per_cell[(agent, scen)] = [score_trial(t, gt) for t in trials]

    agents = sorted({a for a, _ in per_cell})
    scenarios = sorted({s for _, s in per_cell})

    # Table 1 numbers: macro-average across scenarios (each scenario averaged across trials first)
    table1 = {}
    for agent in agents:
        per_scen_means = []
        risk_total = 0
        f1_means, tier_means, calls_means = [], [], []
        for scen in scenarios:
            cell = per_cell.get((agent, scen), [])
            if not cell:
                continue
            per_scen_means.append(_mean([t["tsr"] for t in cell]))
            f1_means.append(_mean([t["macro_f1"] for t in cell]))
            tier_means.append(_mean([t["tier_accuracy"] for t in cell]))
            calls_means.append(_mean([t["calls_per_email"] for t in cell]))
            # risk_errors: sum across scenarios, averaged across trials
            risk_total += _mean([t["risk_errors"] for t in cell])
        table1[agent] = {
            "tsr": _mean(per_scen_means),
            "macro_f1": _mean(f1_means),
            "tier_acc": _mean(tier_means),
            "risk_err_total": risk_total,
            "calls_per_email": _mean(calls_means),
        }

    # Table 2: TSR by tier (mean across scenarios within tier × trials within scenario)
    table2 = {}
    for agent in agents:
        by_tier: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
        for scen in scenarios:
            cell = per_cell.get((agent, scen), [])
            if not cell:
                continue
            tier = _TIER_OF_SCENARIO.get(scen, "")
            if tier in by_tier:
                by_tier[tier].append(_mean([t["tsr"] for t in cell]))
        table2[agent] = {tier: _mean(vs) for tier, vs in by_tier.items()}

    return {"table1": table1, "table2": table2, "per_cell": per_cell}


def render_markdown(run_dir: Path, agg: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark report — {run_dir.name}")
    lines.append("")
    lines.append("## Table 1: Overall performance")
    lines.append("")
    lines.append("| Agent | TSR | Macro-F1 | Tier Acc. | Risk Err. | LLM Calls/Email |")
    lines.append("|---|---|---|---|---|---|")
    for agent, row in agg["table1"].items():
        lines.append(
            f"| {_AGENT_LABELS.get(agent, agent)} "
            f"| {row['tsr']:.2f} | {row['macro_f1']:.2f} | {row['tier_acc']:.2f} "
            f"| {row['risk_err_total']:.1f} | {row['calls_per_email']:.1f} |"
        )

    lines.append("")
    lines.append("## Table 2: TSR by difficulty tier")
    lines.append("")
    lines.append("| Agent | Easy | Medium | Hard |")
    lines.append("|---|---|---|---|")
    for agent, row in agg["table2"].items():
        lines.append(
            f"| {_AGENT_LABELS.get(agent, agent)} "
            f"| {row['easy']:.2f} | {row['medium']:.2f} | {row['hard']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python3 -m benchmarks.scoring.aggregate <run_dir>", file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 1
    agg = aggregate(run_dir)
    report = render_markdown(run_dir, agg)
    out = run_dir / "report.md"
    out.write_text(report)
    print(f"wrote {out}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
