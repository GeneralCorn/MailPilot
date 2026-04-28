import json

from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "database" / "emails.json"


def _load() -> list[dict]:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []


def _save(emails: list[dict]):
    DATA_FILE.write_text(json.dumps(emails, indent=2, default=str))


def get_email_body(email_id: str) -> dict | None:
    """Look up immutable email fields from emails.json by id. None if not found."""
    for e in _load():
        if e.get("id") == email_id:
            return e
    return None


def load_inbox() -> list[dict]:
    """emails.json bodies merged with sqlite email_state for UI rendering."""
    from . import persistence
    emails = _load()
    if not emails:
        return []
    persistence.init_db()
    states = persistence.list_email_states([e["id"] for e in emails if e.get("id")])
    return [{**e, **states.get(e.get("id", ""), {})} for e in emails]
