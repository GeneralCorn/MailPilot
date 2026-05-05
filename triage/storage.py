import json

from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "database"
DATA_FILE = _DB_DIR / "emails.json"
TRACE_FILE = _DB_DIR / "traces.json"

def _load() -> list[dict]:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []

def _save(emails: list[dict]):
    DATA_FILE.write_text(json.dumps(emails, indent=2, default=str))

def _load_traces() -> list[dict]:
    if TRACE_FILE.exists():
        return json.loads(TRACE_FILE.read_text())
    return []

def _save_trace(trace: dict):
    traces = _load_traces()
    traces.append(trace)
    TRACE_FILE.write_text(json.dumps(traces, indent=2, default=str))
