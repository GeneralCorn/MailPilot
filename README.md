# MailPilot

Policy-aware email triage agent. Classifies, prioritizes, and executes actions on emails using a multi-stage LLM pipeline (Router, Evaluator, Ranker, Worker).

## Setup

```bash
cd MailPilot
pip install django
python3 manage.py migrate
python3 manage.py seed
python3 manage.py runserver
```
