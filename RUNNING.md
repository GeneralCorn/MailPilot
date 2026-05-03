# Running MailPilot

## Prerequisites

- Python 3.11+
- A Google Cloud project with **Gmail API** and **Google Calendar API** enabled
- A DeepSeek API key (default), or an Anthropic API key

## One-time setup

### 1. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install django pydantic anthropic google-api-python-client google-auth-oauthlib
```

### 2. Create OAuth credentials

In Google Cloud Console:

1. APIs & Services → Library → enable **Gmail API** and **Google Calendar API**
2. APIs & Services → OAuth consent screen → External → add your Gmail as a **Test user**
3. APIs & Services → Credentials → Create Credentials → **OAuth client ID** → **Desktop app**
4. Download the JSON and save it as `credentials.json` at the repo root

### 3. Set the LLM key

By default the agents call DeepSeek's `deepseek-chat` model via its Anthropic-compatible endpoint:

```bash
export DEEPSEEK_API_KEY=sk-...
```

To use upstream Anthropic Claude (`claude-sonnet-4-6`) instead:

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the web UI

```bash
.venv/bin/python manage.py runserver
```

Open http://127.0.0.1:8000/ .

The first click on **Import Gmail** opens a browser tab to authorize. Pick the Google account whose inbox you want to triage. `token.json` is written next to `credentials.json` and reused on later runs.

## How to use it

| Button | What it does |
|---|---|
| **Import Gmail** | Fetch the 20 most recent emails into the local store |
| **Triage This Email** | Run the full pipeline (router → evaluator → ranker → worker) on the selected email — 5–15 s of LLM work |
| **Run Triage** | Run the pipeline on every imported email (~1 min for 25 emails) |

After triage, each email row shows its category / priority / status. The right-side **Task Queue** lists emails by attention priority (`awaiting_approval` first, then `flagged`, `partial_done`, `pending`, `done`).

## Approving proposed actions

For Calendar event creation and RSVP / confirmation Gmail sends, the Worker does **not** execute on its own. Instead it queues the action as a proposal. The selected email shows an orange **Proposals** panel above the body with:

- Editable parameters (start/end time, location, recipient, body, …)
- **Approve** — runs the real Google API call
- **Reject** — discards the proposal

Approving a `send_email` is idempotent per email + decision: re-clicking won't double-send.

## Re-authorize with a different Google account

```bash
rm token.json
```

The next Google API call (e.g. clicking Import Gmail) triggers a new OAuth flow.

## Storage layout

- `database/emails.json` — imported email bodies. Only `Import Gmail` writes here.
- `database/mailpilot.sqlite3` — pipeline state, snapshots, run history, per-email mutable fields (category / status / priority / draft / calendar / proposals / external action history).
- `token.json` — Google OAuth token (gitignored).
- `credentials.json` — OAuth client secret (gitignored).

To reset all pipeline state and re-triage from scratch (email bodies are kept):

```bash
rm database/mailpilot.sqlite3*
```

## Cost

DeepSeek `deepseek-chat` is the default and is materially cheaper than Sonnet — a 25-email batch typically costs a few cents.

If you switch to Anthropic Sonnet 4.6 (`LLM_PROVIDER=anthropic`), with prompt caching a 25-email batch is roughly **$0.20–0.50** on first run and **~$0.10** if re-run within 5 minutes (cache hits).
