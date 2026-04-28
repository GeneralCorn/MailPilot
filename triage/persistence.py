from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .schemas import State, Status

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "mailpilot.sqlite3"

STAGES = ("input", "router", "evaluator", "ranker", "worker")

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
)

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()
_conn_pid: int | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Create tables if missing. Safe to call repeatedly."""
    conn = get_conn(db_path)
    with conn:
        for stmt in _DDL:
            conn.execute(stmt)
    return conn


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Per-process singleton connection. Pass db_path only on first call (or after reset_conn)."""
    global _conn, _conn_pid
    import os
    pid = os.getpid()
    with _conn_lock:
        if _conn is not None and _conn_pid == pid:
            return _conn
        target = db_path or DB_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(target), check_same_thread=False, isolation_level=None)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn_pid = pid
        return _conn


def reset_conn() -> None:
    """For tests: drop the cached connection so the next get_conn opens a fresh one."""
    global _conn, _conn_pid
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                pass
        _conn = None
        _conn_pid = None


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
