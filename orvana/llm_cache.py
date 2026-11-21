"""Tiny persistent cache for LLM answers (SQLite).

Key = sha256(model digest + prompt version + input). Benefits: duplicate texts are classified once
(~16% of tickets are exact duplicates), re-runs are reproducible, and an interrupted run resumes.
"""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .config import settings


class LLMCache:
    def __init__(self, path: Path | None = None):
        path = path or settings.state_dir / "llm_cache.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # one connection shared by worker threads
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
