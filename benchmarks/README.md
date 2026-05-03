# MailPilot Benchmark

10 inbox scenarios (Easy E1–E3, Medium M1–M4, Hard H1–H3, totaling 70 emails) with hand-curated ground truth. Compares MailPilot against three baselines under a 4-gate Task Success Rate.

## Layout

```
scenarios/        E1–H3 input emails (10 files)
ground_truth/     E1–H3 expected category/priority/action/needs_review per email + tool calls
baselines/        rule_based, single_prompt, no_evaluator, mailpilot_full
scoring/          tsr, metrics, tool_actions, errors, aggregate, validate_gt
human_eval/       template csv, make_csv, irr (Cohen's κ)
results/          run output (gitignored)
run_benchmark.py  CLI entry
```

## Quick start

Pre-flight:
```
python3 -m benchmarks.scoring.validate_gt
.venv/bin/python -m pytest benchmarks/ triage/
```

Run one agent across the full benchmark, 3 trials per scenario:
```
.venv/bin/python benchmarks/run_benchmark.py --agent rule       --difficulty all --trials 3 --run-id paper
.venv/bin/python benchmarks/run_benchmark.py --agent single     --difficulty all --trials 3 --run-id paper --max-cost-usd 5
.venv/bin/python benchmarks/run_benchmark.py --agent no_eval    --difficulty all --trials 3 --run-id paper --max-cost-usd 5
.venv/bin/python benchmarks/run_benchmark.py --agent mailpilot  --difficulty all --trials 3 --run-id paper --max-cost-usd 10
```

Reusing the same `--run-id` lets all four agents land in `results/paper/`. The runner writes `results/paper/report.md` automatically with paper Table 1 + Table 2.

To regenerate the report from existing trial files:
```
.venv/bin/python -m benchmarks.scoring.aggregate benchmarks/results/paper
```

## Output shape

```
results/<run-id>/
  <agent>/<scenario>/trial_<K>.json    # per-trial: predictions, usage, model, git commit, timestamp
  report.md                             # Table 1 (overall) + Table 2 (by tier)
```

Each trial JSON records `model`, `git_commit`, `timestamp`, `temperature` for reproducibility, plus `usage.calls / input_tokens / output_tokens / estimated_cost_usd`.

## Scoring gates (TSR)

A scenario passes (TSR=1) only if all four gates pass; reported TSR is the macro-average across scenarios.

1. **schema_valid** — every per-email row has the required keys with values from the canonical enums (`benchmarks/scoring/metrics.py` `CATEGORIES / PRIORITIES / ACTIONS`).
2. **no_unsafe_risk** — every Risk-category GT row got an action in `{escalate, flag}`.
3. **tool_action ≥ 0.8** — `expected_tool_calls` satisfied (tool name + required args). Mapping between GT names (`calendar.create_event`) and worker output names (`calendar`) lives in `scoring/tool_actions.py`.
4. **tier_accuracy ≥ 0.8** — predicted `priority` field matches GT.

## Human evaluation

Generate a blank rating CSV from a finished agent run:
```
.venv/bin/python -m benchmarks.human_eval.make_csv benchmarks/results/paper --agent mailpilot
```
Two raters fill `rater_a` / `rater_b` columns (1–5) for each (email × criterion) row across the four criteria: explanation_clarity, signal_grounding, action_coherence, batch_consistency.

Then compute inter-rater agreement:
```
.venv/bin/python -m benchmarks.human_eval.irr benchmarks/results/paper/mailpilot_ratings.csv
```

## LLM backend

Default is `deepseek-chat` via DeepSeek's Anthropic-compatible endpoint. Set `DEEPSEEK_API_KEY`. To use upstream Claude instead, `export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...`. See `triage/agent/_client.py`.

Pricing per 1M tokens (used for `--max-cost-usd` cap) lives at the top of `run_benchmark.py`.
