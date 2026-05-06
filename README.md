# MailPilot

Policy-aware batch email triage agent. A four-stage LLM pipeline (Router → Evaluator → Ranker → Worker) classifies, prioritizes, and acts on inbox emails. Runs on `deepseek-chat` by default (DeepSeek's Anthropic-compatible endpoint); `claude-sonnet-4-6` is also supported.

The repo contains:
- `triage/` — the agent pipeline (Django app)
- `mailpilot/` — Django project skeleton
- `benchmarks/` — 10-scenario benchmark, baselines, scoring, and the runner that produces the two summary tables (overall performance + TSR by difficulty tier)
- `evaluation/` — older router/evaluator-only diagnostic harness (not used for the headline numbers)

## Setup

Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

LLM key — pick one:

```bash
export DEEPSEEK_API_KEY=sk-...                 # default
# or
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

For the live web UI you also need Gmail + Calendar OAuth credentials (`credentials.json` at repo root). The benchmark harness does not require them.

## Run the agent

Web UI:

```bash
.venv/bin/python manage.py runserver
# open http://127.0.0.1:8000/
```

The first **Import Gmail** click triggers OAuth. **Triage This Email** runs one email through the full pipeline; **Run Triage** runs all imported emails. Calendar/RSVP actions queue as proposals — approve in the UI to actually fire the Google API call.

State lives in `database/mailpilot.sqlite3`. Wipe it (`rm database/mailpilot.sqlite3*`) to re-triage from scratch.

## Reproduce the paper results

`benchmarks/run_benchmark.py` runs each agent on every scenario 3 times and aggregates two markdown tables into `report.md`:

- **Overall performance** — per agent: TSR, macro-F1, tier accuracy, risk-handling error count, LLM calls per email.
- **TSR by difficulty tier** — per agent × {Easy, Medium, Hard}.

Pre-flight (validates the 10 ground-truth files):

```bash
.venv/bin/python -m benchmarks.scoring.validate_gt
```

Run all four agents into one shared run directory:

```bash
.venv/bin/python benchmarks/run_benchmark.py --agent rule       --difficulty all --trials 3 --run-id eval1
.venv/bin/python benchmarks/run_benchmark.py --agent single     --difficulty all --trials 3 --run-id eval1 --max-cost-usd 5
.venv/bin/python benchmarks/run_benchmark.py --agent no_eval    --difficulty all --trials 3 --run-id eval1 --max-cost-usd 5
.venv/bin/python benchmarks/run_benchmark.py --agent mailpilot  --difficulty all --trials 3 --run-id eval1 --max-cost-usd 10
```

Aggregate the trial JSONs into the report:

```bash
.venv/bin/python -m benchmarks.scoring.aggregate benchmarks/results/eval1
cat benchmarks/results/eval1/report.md
```

Per-trial JSONs under `benchmarks/results/<run-id>/<agent>/<scenario>/trial_K.json` carry full provenance: `model`, `git_commit`, `timestamp`, `temperature`, `usage` (calls / input_tokens / output_tokens / estimated_cost_usd).

End-to-end DeepSeek cost for the full 4×10×3 = 120 trials is under $0.10 at current `deepseek-v4-flash` pricing.

See [`benchmarks/README.md`](benchmarks/README.md) for the scoring rubric (4-gate soft TSR) and how to add a new agent or scenario.
