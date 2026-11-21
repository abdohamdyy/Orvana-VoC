"""Where chat data lives, and what is allowed to reach the disk.

- Conversations live in MEMORY only (SessionStore): idle sessions are dropped after chat_session_ttl_s,
  and no transcript is ever written to disk. The LLM is called with use_cache=False for the same reason.
- When a case becomes a ticket, ONE row is written (TicketSink): the structured record plus the
  complaint text after redaction (emails, card and phone numbers, and health details masked: the
  organisation stores no health information). Safety and category are decided BEFORE redaction.
"""
import csv
import re
import threading
import time
from datetime import datetime

from ..config import settings
from ..normalize import clean_message
from ..report.mailer import csv_safe

_HEALTH = re.compile(
    r"مستشف\w*|دكتور\w*|طبيب\w*|اتسمم\w*|تسمم\w*|حساسي\w*|عيان\w*|تعبان\w*|اتعب\w*|تعب\w*|مرض\w*|دوا\b|دواء|علاج\w*|اسهال|"
    r"ترجيع|استفراغ|اسعاف|طوارئ|"
    r"\b(?:hospital\w*|doctor\w*|clinic\w*|poison\w*|allerg\w*|sick\w*|ill|illness|vomit\w*|diarrh\w*|medicine|medication|"
    r"emergency|ambulance|mostashfa|doktor|ta3ban|3ayan|tasamom|et3ab)\b", re.IGNORECASE)


def redact_for_storage(text: str) -> str:
    """The only form of customer text that may be written to disk."""
    return _HEALTH.sub("[HEALTH]", clean_message(text).text)


class TicketSink:
    """One row per chat ticket in state/chat/tickets_<date>.csv: the incoming/ columns (channel = chat), so
    the weekly job can read chat tickets like any other channel, plus the structured fields."""
    FIELDS = ["ticket_id", "customer_id", "created_at", "channel", "message_text", "category",
              "product_mentioned", "store_mentioned", "urgency"]

    def __init__(self, directory=None):
        self.dir = directory or settings.state_dir / "chat"
        self.lock = threading.Lock()

    def write(self, st, case, record) -> None:
        now = datetime.now()
        row = {"ticket_id": record.ticket_id, "customer_id": st.customer_id or "", "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
               "channel": "chat", "message_text": redact_for_storage(case.text), "category": record.category.value,
               "product_mentioned": record.product_mentioned or "", "store_mentioned": record.store_mentioned or "",
               "urgency": record.urgency}
        path = self.dir / f"tickets_{now:%Y-%m-%d}.csv"
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: csv_safe(v) for k, v in row.items()})


class SessionStore:
    """In-memory sessions with a TTL and a size cap. Each session has its own lock, so one customer's
    slow LLM turn never blocks other customers, and two requests of one session never interleave."""

    def __init__(self, ttl_s: int | None = None, max_sessions: int | None = None):
        self.ttl = ttl_s or settings.chat_session_ttl_s
        self.max = max_sessions or settings.chat_max_sessions
        self._items: dict[str, list] = {}  # session_id -> [state, lock, last_seen]
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None, factory):
        """(state, lock) for an existing live session, or a new one made by factory()."""
        now = time.monotonic()
        with self._lock:
            for sid in [s for s, (_, _, seen) in self._items.items() if now - seen > self.ttl]:
                del self._items[sid]  # idle sessions are forgotten
            item = self._items.get(session_id or "")
            if item is None:
                while len(self._items) >= self.max:  # full: evict the least recently used
                    del self._items[min(self._items, key=lambda s: self._items[s][2])]
                state = factory()
                item = self._items[state.session_id] = [state, threading.Lock(), now]
            item[2] = now
            return item[0], item[1]

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._items)
