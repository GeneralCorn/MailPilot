# MailPilot

Policy-aware email triage agent. Classifies, prioritizes, and executes actions on emails using a multi-stage LLM pipeline (Router, Evaluator, Ranker, Worker).

## Setup

```bash
cd MailPilot
pip install django pydantic google-api-python-client google-auth-oauthlib
python3 manage.py runserver
```

## Gmail Import

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com)
2. Save as `credentials.json` next to `manage.py`
3. Add your Gmail as a test user in OAuth consent screen
4. Click **Import Gmail** in the UI
