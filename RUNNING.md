# Running MailPilot with Real APIs

## Prerequisites

- Python 3.11+
- A Google Cloud project with **Gmail API** and **Google Calendar API** enabled
- An Anthropic API key

## One-time setup

### 1. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install django pydantic anthropic google-api-python-client google-auth-oauthlib pytest
```

### 2. Create OAuth credentials

In Google Cloud Console:

1. APIs & Services → Library → enable **Gmail API** and **Google Calendar API**
2. APIs & Services → OAuth consent screen → External → add your Gmail as a **Test user**
3. APIs & Services → Credentials → Create Credentials → **OAuth client ID** → **Desktop app**
4. Download the JSON and save it as `credentials.json` at the repo root

### 3. Set the Anthropic key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Mock mode vs real mode

| Mode | How | Effect |
|---|---|---|
| Mock | `export MAILPILOT_MOCK_TOOLS=1` | Calendar / send_email do **not** call real APIs; only SQLite is written. LLM calls still run. |
| Real | `unset MAILPILOT_MOCK_TOOLS` | Calendar events are really created; emails are really sent. |

Always test in mock mode first.

## Run the web UI

```bash
.venv/bin/python manage.py runserver
```

Open http://127.0.0.1:8000/ .

The first click on **Import Gmail** opens a browser tab to authorize. Pick the Google account whose inbox you want to triage. `token.json` is written next to `credentials.json` and reused.

| Button | What it does |
|---|---|
| Import Gmail | Fetch the 20 most recent emails into `database/emails.json` |
| Triage This Email | Run the full pipeline on the selected email (5–15s) |
| Run Triage | Run the pipeline on every imported email (~1 min for 25 emails) |

After triage, each email's category / priority / status appears on its row, and the right-side Task Queue lists what was done with reasons.

## Smoke-test real APIs without the LLM

These call the tools directly, no LLM involved.

**Calendar event** (creates a real event 1 hour from now on the authorized account's primary calendar):

```bash
unset MAILPILOT_MOCK_TOOLS
.venv/bin/python -c "
import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mailpilot.settings')
django.setup()
from datetime import datetime, timedelta, timezone
from triage.tools.actions import create_calendar_event
start = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
end   = (datetime.now(timezone.utc) + timedelta(hours=1, minutes=30)).replace(microsecond=0).isoformat()
print(create_calendar_event(email_id='smoke', title='MailPilot smoke test', start_time=start, end_time=end))
"
```

**Gmail send** (sends a real email — use your own address as the recipient):

```bash
unset MAILPILOT_MOCK_TOOLS
.venv/bin/python -c "
import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mailpilot.settings')
django.setup()
from triage import gmail
print(gmail.send_email(to='YOUR_GMAIL_ADDRESS', subject='MailPilot smoke test', body='Hello.'))
"
```

Verify the event in Calendar / the email in Gmail. The authorized account is the one whose `token.json` you created — it may be different from the Google account your browser is currently signed into.

## Run the test suite

```bash
.venv/bin/python -m pytest
```

Tests use mock mode by default (set in `triage/conftest.py`). They don't need `credentials.json` or a valid `ANTHROPIC_API_KEY`.

## Re-authorize with a different Google account

```bash
rm token.json
```

The next API call triggers a new OAuth flow.

## Storage layout

- `database/emails.json` — imported email bodies. Only `Import Gmail` writes here.
- `database/mailpilot.sqlite3` — pipeline state, snapshots, run history, per-email mutable fields (category / status / priority / draft / calendar / etc.).
- `token.json` — Google OAuth token (gitignored).
- `credentials.json` — OAuth client secret (gitignored).

To wipe pipeline state and re-run from scratch (email bodies are kept):

```bash
rm database/mailpilot.sqlite3*
```

## Cost note

With Sonnet 4.6 + prompt caching, a 25-email batch is roughly **$0.20–0.50** first run and **~$0.10** if re-run within 5 minutes (cache hits). Mock mode still pays for LLM calls; only external side-effects are skipped.
