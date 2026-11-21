"""Persistent state that makes the weekly job idempotent (SQLite in state/, a Docker volume).

files   one row per incoming file: its hash and whether its tickets were classified
tickets one row per ticket_id (PRIMARY KEY => duplicate rows in a file are stored once)
emails  one row per week (PRIMARY KEY => an email can be claimed only once)

Email state machine:  (none) -> sending -> sent
A row stuck in 'sending' means we crashed during SMTP; we do NOT resend automatically
(a duplicate email to management is worse than a delayed one); the run logs an alert instead
and `run_weekly.py --resend WEEK` resolves it by hand.
"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    name TEXT PRIMARY KEY, sha256 TEXT NOT NULL, week_key TEXT NOT NULL,
    status TEXT NOT NULL, rows_received INTEGER, processed_at TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY, week_key TEXT NOT NULL, created_at TEXT, channel TEXT,
    customer_id TEXT, message_text TEXT, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS emails (
    week_key TEXT PRIMARY KEY, status TEXT NOT NULL, message_id TEXT, eml_path TEXT, updated_at TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, path: Path | None = None):
        path = path or settings.state_dir / "state.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)  # autocommit; explicit BEGIN where needed
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- files
    def file_row(self, name: str):
        return self.db.execute("SELECT sha256, week_key, status FROM files WHERE name=?", (name,)).fetchone()

    def save_tickets(self, name: str, sha: str, week_key: str, rows_received: int, rows: list[dict]) -> None:
        """Store classified tickets and mark the file processed, atomically."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.executemany(
                "INSERT OR IGNORE INTO tickets VALUES (:ticket_id, :week_key, :created_at, :channel, "
                ":customer_id, :message_text, :record)", rows)
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, 'processed', ?, ?)",
                            (name, sha, week_key, rows_received, now()))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def week_tickets(self, week_key: str) -> list[dict]:
        cur = self.db.execute("SELECT ticket_id, created_at, channel, customer_id, message_text, record "
                              "FROM tickets WHERE week_key=?", (week_key,))
        out = []
        for tid, created, channel, cust, text, rec in cur.fetchall():
            out.append({"ticket_id": tid, "created_at": created, "channel": channel, "customer_id": cust,
                        "message_text": text, **json.loads(rec)})
        return out

    def file_meta(self, week_key: str):
        return self.db.execute("SELECT name, rows_received FROM files WHERE week_key=?", (week_key,)).fetchone()

    def previous_week_key(self, week_key: str) -> str | None:
        row = self.db.execute("SELECT MAX(week_key) FROM files WHERE week_key < ?", (week_key,)).fetchone()
        return row[0] if row else None

    # ---------------------------------------------------------------- emails
    def email_status(self, week_key: str) -> str | None:
        row = self.db.execute("SELECT status FROM emails WHERE week_key=?", (week_key,)).fetchone()
        return row[0] if row else None

    def claim_email(self, week_key: str, message_id: str) -> bool:
        """True only for the first caller; the PRIMARY KEY makes a second claim fail."""
        try:
            self.db.execute("INSERT INTO emails VALUES (?, 'sending', ?, NULL, ?)", (week_key, message_id, now()))
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_sent(self, week_key: str, eml_path: str) -> None:
        self.db.execute("UPDATE emails SET status='sent', eml_path=?, updated_at=? WHERE week_key=?",
                        (eml_path, now(), week_key))

    def release_email(self, week_key: str) -> None:
        """Used when sending failed for sure (SMTP refused): allow a later retry."""
        self.db.execute("DELETE FROM emails WHERE week_key=? AND status='sending'", (week_key,))


@contextmanager
def run_lock(name: str = "weekly", stale_after_s: int = 3 * 3600):
    """Prevent two overlapping runs (e.g. the scheduler and a manual run). A lock older than
    `stale_after_s` is assumed to belong to a crashed run and is taken over."""
    path = settings.state_dir / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > stale_after_s:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"another run holds {path}; try again later") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)
