import json

from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "database" / "emails.json"

def _load() -> list[dict]:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []

def _save(emails: list[dict]):
    DATA_FILE.write_text(json.dumps(emails, indent=2, default=str))

