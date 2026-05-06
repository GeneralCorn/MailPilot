from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import State, Status

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "mailpilot.sqlite3"

STAGES = ("input", "router", "evaluator", "ranker", "worker")

# fields that get materialized as plain columns; everything else (lists/dicts) goes JSON-encoded
_STATE_SCALAR_FIELDS = ("category", "priority", "priority_rank", "status", "flagged", "folder", "summary")
_STATE_JSON_FIELDS = ("draft_reply", "calendar_event", "escalations", "notes", "external_actions", "proposed_actions")
_STATE_LIST_FIELDS = ("escalations", "notes", "external_actions", "proposed_actions")
_STATE_ALL_FIELDS = _STATE_SCALAR_FIELDS + _STATE_JSON_FIELDS

_STATE_DEFAULTS: dict[str, Any] = {
    "category": "unclassified",
    "priority": "",
    "priority_rank": 9999,
    "status": "",
    "flagged": False,
    "folder": "",
    "summary": "",
    "draft_reply": None,
    "calendar_event": None,
    "escalations": [],
    "notes": [],
    "external_actions": [],
    "proposed_actions": [],
}

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS email_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        email_id TEXT,
        stage TEXT NOT NULL,
        state_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_snap_run_email_stage ON email_snapshots(run_id, email_id, stage)",
    """
    CREATE TABLE IF NOT EXISTS processed_emails (
        email_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        status TEXT NOT NULL,
        finished_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS email_state (
        email_id TEXT PRIMARY KEY,
        category TEXT,
        priority TEXT,
        priority_rank INTEGER,
        status TEXT,
        flagged INTEGER DEFAULT 0,
        folder TEXT,
        summary TEXT,
        draft_reply TEXT,
        calendar_event TEXT,
        escalations TEXT,
        notes TEXT,
        external_actions TEXT,
        proposed_actions TEXT,
        updated_at TEXT NOT NULL
    )
    """,
)

# columns added after initial release; ALTER TABLE on existing DBs
_ADDED_EMAIL_STATE_COLUMNS = (
    ("proposed_actions", "TEXT"),
)

_conn_storage = threading.local()
_db_path_lock = threading.Lock()
_db_path: Path | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path | None = None, *, migrate_from_json: bool = True) -> sqlite3.Connection:
    """Create tables if missing. Safe to call repeatedly. Optionally backfills email_state from emails.json."""
    conn = get_conn(db_path)
    with conn:
        for stmt in _DDL:
            conn.execute(stmt)
        for col, decl in _ADDED_EMAIL_STATE_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE email_state ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists
    if migrate_from_json:
        migrate_emails_json_to_state()
    return conn


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Thread-local connection. SQLite handles cross-thread synchronization via WAL."""
    global _db_path
    with _db_path_lock:
        if db_path is not None:
            _db_path = db_path
        target = _db_path or DB_PATH

    existing = getattr(_conn_storage, "conn", None)
    if existing is not None:
        existing_path = getattr(_conn_storage, "path", None)
        if existing_path == str(target):
            return existing
        try:
            existing.close()
        except sqlite3.Error:
            pass

    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _conn_storage.conn = conn
    _conn_storage.path = str(target)
    return conn


def reset_conn() -> None:
    """Drop this thread's connection and forget any pinned db path. Used by tests."""
    global _db_path
    existing = getattr(_conn_storage, "conn", None)
    if existing is not None:
        try:
            existing.close()
        except sqlite3.Error:
            pass
    if hasattr(_conn_storage, "conn"):
        del _conn_storage.conn
    if hasattr(_conn_storage, "path"):
        del _conn_storage.path
    with _db_path_lock:
        _db_path = None


def start_run(run_id: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, started_at, status) VALUES (?, ?, ?)",
        (run_id, _utcnow(), "running"),
    )


def finish_run(run_id: str, status: str = "done") -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE runs SET finished_at=?, status=? WHERE run_id=?",
        (_utcnow(), status, run_id),
    )


def snapshot(run_id: str, email_id: str | None, stage: str, state: State) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO email_snapshots (run_id, email_id, stage, state_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, email_id, stage, state.model_dump_json(), _utcnow()),
    )


def latest_snapshot(run_id: str, email_id: str | None) -> tuple[str, State] | None:
    """Latest snapshot for a (run, email) — email_id=None for whole-run stages like ranker."""
    conn = get_conn()
    if email_id is None:
        row = conn.execute(
            "SELECT stage, state_json FROM email_snapshots "
            "WHERE run_id=? AND email_id IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT stage, state_json FROM email_snapshots "
            "WHERE run_id=? AND email_id=? "
            "ORDER BY id DESC LIMIT 1",
            (run_id, email_id),
        ).fetchone()
    if row is None:
        return None
    stage, state_json = row
    return stage, State.model_validate_json(state_json)


def mark_processed(email_id: str, run_id: str, status: Status | str) -> None:
    conn = get_conn()
    val = status.value if isinstance(status, Status) else str(status)
    conn.execute(
        "INSERT INTO processed_emails (email_id, run_id, status, finished_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(email_id) DO UPDATE SET run_id=excluded.run_id, status=excluded.status, finished_at=excluded.finished_at",
        (email_id, run_id, val, _utcnow()),
    )


def is_processed(email_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM processed_emails WHERE email_id=? AND status=?",
        (email_id, Status.DONE.value),
    ).fetchone()
    return row is not None


def unfinished_runs() -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT run_id FROM runs WHERE finished_at IS NULL ORDER BY started_at"
    ).fetchall()
    return [r[0] for r in rows]


def _row_to_state(row: tuple) -> dict[str, Any]:
    cols = ("email_id", "category", "priority", "priority_rank", "status", "flagged",
            "folder", "summary", "draft_reply", "calendar_event", "escalations",
            "notes", "external_actions", "proposed_actions", "updated_at")
    out = dict(zip(cols, row))
    out["flagged"] = bool(out.get("flagged") or 0)
    for f in _STATE_JSON_FIELDS:
        raw = out.get(f)
        if raw is None or raw == "":
            out[f] = list(_STATE_DEFAULTS[f]) if isinstance(_STATE_DEFAULTS[f], list) else _STATE_DEFAULTS[f]
        else:
            out[f] = json.loads(raw)
    for f in _STATE_SCALAR_FIELDS:
        if out.get(f) is None:
            out[f] = _STATE_DEFAULTS[f]
    return out


def get_email_state(email_id: str) -> dict[str, Any]:
    """Return the email_state row as a dict; defaults filled in for missing rows or columns."""
    conn = get_conn()
    row = conn.execute(
        "SELECT email_id, category, priority, priority_rank, status, flagged, "
        "folder, summary, draft_reply, calendar_event, escalations, notes, "
        "external_actions, proposed_actions, updated_at FROM email_state WHERE email_id=?",
        (email_id,),
    ).fetchone()
    if row is None:
        return {"email_id": email_id, **{k: (list(v) if isinstance(v, list) else v) for k, v in _STATE_DEFAULTS.items()}}
    return _row_to_state(row)


def list_email_states(email_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
    conn = get_conn()
    if email_ids is None:
        rows = conn.execute(
            "SELECT email_id, category, priority, priority_rank, status, flagged, "
            "folder, summary, draft_reply, calendar_event, escalations, notes, "
            "external_actions, proposed_actions, updated_at FROM email_state"
        ).fetchall()
    elif not email_ids:
        return {}
    else:
        placeholders = ",".join(["?"] * len(email_ids))
        rows = conn.execute(
            f"SELECT email_id, category, priority, priority_rank, status, flagged, "
            f"folder, summary, draft_reply, calendar_event, escalations, notes, "
            f"external_actions, proposed_actions, updated_at FROM email_state WHERE email_id IN ({placeholders})",
            tuple(email_ids),
        ).fetchall()
    return {r[0]: _row_to_state(r) for r in rows}


def update_email_state(email_id: str, **fields: Any) -> None:
    """Upsert: insert row if missing, otherwise patch the given fields. JSON-encodes list/dict fields."""
    if not fields:
        return
    unknown = set(fields) - set(_STATE_ALL_FIELDS)
    if unknown:
        raise ValueError(f"unknown email_state fields: {sorted(unknown)}")

    encoded: dict[str, Any] = {}
    for k, v in fields.items():
        if k in _STATE_JSON_FIELDS:
            encoded[k] = json.dumps(v) if v is not None else None
        elif k == "flagged":
            encoded[k] = int(bool(v))
        else:
            encoded[k] = v

    conn = get_conn()
    cols = list(encoded.keys())
    placeholders = ",".join(["?"] * (len(cols) + 2))
    insert_cols = ["email_id"] + cols + ["updated_at"]
    insert_vals = [email_id] + [encoded[c] for c in cols] + [_utcnow()]
    update_clause = ", ".join([f"{c}=excluded.{c}" for c in cols] + ["updated_at=excluded.updated_at"])
    conn.execute(
        f"INSERT INTO email_state ({','.join(insert_cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(email_id) DO UPDATE SET {update_clause}",
        tuple(insert_vals),
    )


def append_email_list_field(email_id: str, field: str, value: Any) -> None:
    """Append to a JSON-list field (escalations / notes / external_actions / proposed_actions)."""
    if field not in _STATE_LIST_FIELDS:
        raise ValueError(f"{field!r} is not a list field; expected one of {_STATE_LIST_FIELDS}")
    current = get_email_state(email_id)[field]
    update_email_state(email_id, **{field: [*current, value]})


def has_pending_proposals(email_id: str) -> bool:
    return bool(get_email_state(email_id).get("proposed_actions"))


def remove_proposed_action(email_id: str, idx: int) -> dict | None:
    """Remove the i-th proposed action; return the removed dict, or None if idx out of range."""
    current = list(get_email_state(email_id).get("proposed_actions") or [])
    if idx < 0 or idx >= len(current):
        return None
    removed = current.pop(idx)
    update_email_state(email_id, proposed_actions=current)
    return removed


def update_proposed_action(email_id: str, idx: int, parameters: dict) -> dict | None:
    """Merge `parameters` into the i-th proposed action's parameters dict; return the updated dict."""
    current = list(get_email_state(email_id).get("proposed_actions") or [])
    if idx < 0 or idx >= len(current):
        return None
    entry = dict(current[idx])
    entry["parameters"] = {**(entry.get("parameters") or {}), **parameters}
    current[idx] = entry
    update_email_state(email_id, proposed_actions=current)
    return entry


def migrate_emails_json_to_state() -> int:
    """One-shot: copy state-like fields from emails.json into the email_state table.
    No-op if email_state already has rows or emails.json is missing/empty.
    """
    conn = get_conn()
    existing = conn.execute("SELECT COUNT(*) FROM email_state").fetchone()[0]
    if existing:
        return 0

    from .storage import _load
    emails = _load()
    written = 0
    for e in emails:
        eid = e.get("id")
        if not eid:
            continue
        fields = {}
        for f in _STATE_SCALAR_FIELDS:
            if f in e:
                fields[f] = e[f]
        for f in _STATE_JSON_FIELDS:
            if f in e:
                fields[f] = e[f]
        if fields:
            update_email_state(eid, **fields)
            written += 1
    return written
