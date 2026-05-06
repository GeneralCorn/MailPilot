# MailPilot benchmark

10 inbox scenarios (Easy E1–E3 = 5 emails each; Medium M1–M4 = 7 emails each; Hard H1–H3 = 9 emails each), 70 emails total, with hand-curated ground truth. Top-level [`README.md`](../README.md) covers how to run a benchmark from scratch.

## Layout

```
scenarios/        E1–H3 input emails (10 files)
ground_truth/     E1–H3 expected category/priority/action/needs_review + tool calls
baselines/        rule_based, single_prompt, no_evaluator, mailpilot_full
scoring/          tsr, metrics, tool_actions, errors, aggregate, validate_gt
results/          run output (gitignored)
run_benchmark.py  CLI entry
```

## Output shape

```
results/<run-id>/
  <agent>/<scenario>/trial_<K>.json    # per-trial: predictions, usage, model, git commit, timestamp
  report.md                            # Table 1 (overall) + Table 2 (by tier)
```

## Scoring (soft 4-gate TSR)

Each scenario is scored against four gates. The scenario's TSR is `passed_gates / 4` (so 0.0, 0.25, 0.50, 0.75, or 1.0). Reported TSR is the macro-average across scenarios.

1. **schema_valid** — every per-email row has the required keys with values from the canonical enums (`benchmarks/scoring/metrics.py` `CATEGORIES / PRIORITIES / ACTIONS`).
2. **no_unsafe_risk** — every Risk-category GT row got an action in `{escalate, flag}`.
3. **action_accuracy ≥ 0.8** — predicted `action` matches GT for ≥ 80% of emails.
4. **tier_accuracy ≥ 0.8** — predicted `priority` matches GT for ≥ 80% of emails.

`scoring/tool_actions.py` separately scores GT's `expected_tool_calls` (tool name + required args, with mapping between GT names like `calendar.create_event` and worker output names like `calendar`). This is reported alongside but not a gate.

`scoring/errors.py` breaks failures into 5 types (schema / category / ranking / tool-action / risk-handling).

## Adding a scenario

1. Drop a new `XX_name.json` into `scenarios/` (same shape as existing ones).
2. Drop the matching `XX_name.gt.json` into `ground_truth/`.
3. Update `_TIER_OF_SCENARIO` in `scoring/aggregate.py` and `select_scenarios()` in `run_benchmark.py` to know about the new id.
4. Run `python -m benchmarks.scoring.validate_gt` to confirm the GT conforms to canonical enums.

## Adding an agent

1. Create `baselines/<name>.py` exposing `run(emails: list[Message]) -> list[dict]`. Each dict needs `id, category, priority, action, needs_review` (plus optional `proposed_actions`, `external_actions` for tool-action scoring).
2. Register it in `AGENTS` at the top of `run_benchmark.py`.
