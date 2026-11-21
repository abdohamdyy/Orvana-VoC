"""Emerging-issue detection: find complaints about something NEW.

A fixed category list cannot tell management "something new started this week". We look for
terms that appear in many of this week's tickets but NEVER in the history (tickets.csv). Example
(week 2): '5.2' and 'update' appear in 62 tickets and 0 historical ones, revealing a payment bug
after app update 5.2. Language-agnostic, explainable, no model needed. Week 1 gives no alert.
"""
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from ..config import settings
from ..normalize import clean_message, match_key

_TOKEN = re.compile(r"\d+\.\d+|[^\W\d_]{3,}")  # version numbers like 5.2, or words of 3+ letters


def terms(text: str) -> set[str]:
    return set(_TOKEN.findall(match_key(clean_message(text).text)))


@lru_cache(maxsize=1)
def history_terms() -> Counter:
    c = Counter()
    for t in pd.read_csv(settings.data_dir / "tickets.csv", usecols=["message_text"]).message_text:
        c.update(terms(t))
    return c


def emerging_issues(valid: pd.DataFrame, min_tickets: int = 5, max_issues: int = 3) -> list[dict]:
    hist = history_terms()
    tick_terms = {tid: terms(t) for tid, t in zip(valid.ticket_id, valid.message_text)}
    week = Counter(t for ts in tick_terms.values() for t in ts)
    new = {t for t, c in week.items() if c >= min_tickets and hist[t] == 0}
    issues, remaining = [], dict(tick_terms)
    while new and len(issues) < max_issues:
        counts = Counter(t for ts in remaining.values() for t in ts if t in new)
        if not counts or counts.most_common(1)[0][1] < min_tickets:
            break
        anchor = counts.most_common(1)[0][0]
        ids = [tid for tid, ts in remaining.items() if anchor in ts]
        sub = valid[valid.ticket_id.isin(ids)]
        co = Counter(t for tid in ids for t in remaining[tid] if t in new)
        texts = sub.message_text.map(lambda t: clean_message(t).text)
        arabic = texts[texts.str.contains(r"[ء-ي]")]
        latin = texts[~texts.str.contains(r"[ء-ي]")]
        examples = [s.loc[s.str.len().idxmin()] for s in (arabic, latin) if len(s)]
        issues.append({
            "terms": [t for t, _ in co.most_common(8)],
            "tickets": len(ids),
            "share_pct": round(100 * len(ids) / max(len(valid), 1), 1),
            "categories": {k: int(v) for k, v in sub.category.value_counts().items()},
            "first_seen": str(pd.to_datetime(sub.created_at).min()),
            "examples": examples,
        })
        for tid in ids:
            remaining.pop(tid)
        new -= set(co)
    return issues
