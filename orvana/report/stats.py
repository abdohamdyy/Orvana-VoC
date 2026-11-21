"""Every number in the weekly report is computed here, with pandas, from the stored records.
The LLM never computes numbers: it only rephrases the `facts` produced by this module.
"""
from functools import lru_cache

import pandas as pd

from ..config import settings
from ..extractors import (double_charge_amount, find_product, has_safety_issue, is_repeat_complaint, safety_check,
                          split_segments)
from ..normalize import clean_message
from ..schema import CATEGORIES
from .emerging import emerging_issues
from .monitoring import signals


def pct(part: float, whole: float) -> float:
    return round(100.0 * float(part) / float(whole), 1) if whole else 0.0


@lru_cache(maxsize=1)
def historical_shares(weeks: int = 13) -> dict[str, float]:
    """Category shares (%) over the last `weeks` weeks of history (cleaned labels).
    The history is a ~86/week SAMPLE, so only shares are comparable, never counts."""
    t = pd.read_csv(settings.data_dir / "tickets.csv", usecols=["ticket_id", "created_at"])
    m = pd.read_csv(settings.data_dir / "processed" / "messages.csv", usecols=["ticket_id", "primary_clean"])
    h = t.merge(m, on="ticket_id").dropna(subset=["primary_clean"])
    h["created_at"] = pd.to_datetime(h.created_at)
    h = h[h.created_at >= h.created_at.max() - pd.Timedelta(weeks=weeks)]
    return {c: pct((h.primary_clean == c).sum(), len(h)) for c in CATEGORIES}


def safety_report(text: str, product: str | None = None) -> dict | None:
    """Policy §5: ANY complaint about an expired / overheating / burning product goes to the Quality team within
    2 hours, even when it is the second problem in the message (urgency follows the main problem, so such a
    ticket can be urgency 2). 'check' is 'needs_confirmation' when the product type rarely has that hazard."""
    segs = split_segments(clean_message(text).text)
    checks = [(i, safety_check(s, find_product(s) or product)) for i, s in enumerate(segs)]
    checks = [(i, c) for i, c in checks if c]
    if not checks:
        return None
    return {"where": "main issue" if checks[0][0] == 0 else "second issue",
            "check": "confirmed" if any(c == "confirmed" for _, c in checks) else "needs_confirmation"}


def urgency_reason(text: str, category: str, product: str | None = None) -> str:
    c = clean_message(text).text
    first = split_segments(c)[0]
    if is_repeat_complaint(c):
        return "repeat complaint"
    if category == "product_quality" and has_safety_issue(first):
        if safety_check(first, find_product(first) or product) == "needs_confirmation":
            return "safety, to confirm (product rarely has this hazard)"
        return "safety (expired/overheating)"
    if category == "refund_request" and double_charge_amount(first) is not None:
        return f"double charge {double_charge_amount(first)} EGP"
    return ""


def _by_product_store(rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []
    g = (rows.groupby([rows.product_mentioned.fillna("-"), rows.store_mentioned.fillna("-")])
         .size().reset_index(name="count").sort_values("count", ascending=False))
    return g.to_dict("records")


def csv_rows(facts: dict) -> list[dict]:
    """urgent_tickets.csv: every urgency-3 ticket, plus the safety reports that are not urgency 3."""
    return facts["urgent_tickets"] + [s for s in facts["safety_tickets"] if s["urgency"] < 3]


def weekly_facts(week_key: str, rows: list[dict], rows_received: int, prev_rows: list[dict] | None,
                 store_names: dict[str, str]) -> dict:
    df = pd.DataFrame(rows)
    valid = df[df.is_valid]
    prev = pd.DataFrame(prev_rows) if prev_rows else None
    prev_valid = prev[prev.is_valid] if prev is not None else None
    n = len(valid)
    hist = historical_shares()

    categories = []
    for cat in CATEGORIES:
        cnt = int((valid.category == cat).sum())
        row = {"category": cat, "count": cnt, "share_pct": pct(cnt, n), "hist_share_pct": hist.get(cat, 0.0)}
        row["vs_history_pp"] = round(row["share_pct"] - row["hist_share_pct"], 1)
        if prev_valid is not None:
            row["prev_count"] = int((prev_valid.category == cat).sum())
            row["change_vs_prev"] = cnt - row["prev_count"]
        categories.append(row)
    categories.sort(key=lambda r: -r["count"])

    product = lambda p: p if isinstance(p, str) else None
    # every safety report, wherever it sits in the message (policy §5), with its confirmation status
    reports = [safety_report(t, product(p)) for t, p in zip(valid.message_text, valid.product_mentioned)]
    safety = valid[pd.Series([r is not None for r in reports], index=valid.index, dtype=bool)].copy()
    safety["where"] = pd.Series([r["where"] for r in reports if r], index=safety.index, dtype="object")
    safety["check"] = pd.Series([r["check"] for r in reports if r], index=safety.index, dtype="object")
    safety["reason"] = ("safety (" + safety["where"] + ")"
                        + safety["check"].map({"confirmed": "", "needs_confirmation": ", to confirm"}))
    safety["text"] = pd.Series([clean_message(t).text[:160] for t in safety.message_text], index=safety.index, dtype="object")
    safety = safety.sort_values(["check", "created_at"])  # 'confirmed' sorts before 'needs_confirmation'
    confirmed, to_confirm = safety[safety.check == "confirmed"], safety[safety.check == "needs_confirmation"]

    urgent = valid[valid.urgency == 3].copy()
    # explicit dtype: a week with no urgent tickets must still give string columns
    urgent["reason"] = pd.Series([urgency_reason(t, c, product(p)) for t, c, p in
                                  zip(urgent.message_text, urgent.category, urgent.product_mentioned)],
                                 index=urgent.index, dtype="object")
    urgent["text"] = pd.Series([clean_message(t).text[:160] for t in urgent.message_text], index=urgent.index, dtype="object")
    urgent["is_safety"] = urgent.ticket_id.isin(confirmed.ticket_id)
    urgent = urgent.sort_values(["is_safety", "created_at"], ascending=[False, True])  # safety first (2h SLA)

    def top(col, k=5):
        sub = valid.dropna(subset=[col])
        out = []
        for key, g in sub.groupby(col):
            out.append({"key": key, "name": store_names.get(key, key), "count": len(g),
                        "top_category": g.category.value_counts().index[0]})
        return sorted(out, key=lambda r: (-r["count"], r["key"]))[:k]

    staff = valid[valid.category == "staff_behavior"].dropna(subset=["store_mentioned"])
    cust = valid.dropna(subset=["customer_id"]).groupby("customer_id").size()

    facts = {
        "week_key": week_key,
        "period": {"start": str(pd.to_datetime(df.created_at).min().date()), "end": str(pd.to_datetime(df.created_at).max().date())},
        "volume": {
            "rows_received": rows_received,
            "duplicate_rows_removed": rows_received - len(df),
            "invalid_messages": int((~df.is_valid).sum()),
            "injection_attempts": int(df.injection_flag.sum()),
            "valid_tickets": n,
            "prev_valid_tickets": int(len(prev_valid)) if prev_valid is not None else None,
            "change_vs_prev_pct": pct(n - len(prev_valid), len(prev_valid)) if prev_valid is not None and len(prev_valid) else None,
        },
        "categories": categories,
        "urgency": {str(u): int((valid.urgency == u).sum()) for u in (3, 2, 1)},
        "urgent_tickets": urgent[["ticket_id", "created_at", "channel", "category", "product_mentioned",
                                  "store_mentioned", "reason", "text"]].fillna("").to_dict("records"),
        "safety": {"confirmed": len(confirmed), "to_confirm": len(to_confirm),
                   "second_issue": int((safety["where"] == "second issue").sum())},
        "safety_tickets": safety[["ticket_id", "created_at", "channel", "category", "product_mentioned", "store_mentioned",
                                  "urgency", "where", "check", "reason", "text"]].fillna("").to_dict("records"),
        "safety_by_product_store": _by_product_store(confirmed),
        "safety_to_confirm_by_product_store": _by_product_store(to_confirm),
        "top_stores": top("store_mentioned"),
        "top_products": top("product_mentioned"),
        "staff_behavior_by_store": [{"key": k, "name": store_names.get(k, k), "count": int(v)}
                                    for k, v in staff.store_mentioned.value_counts().items()],
        "channels": {k: int(v) for k, v in valid.channel.value_counts().items()},
        "multi_issue_tickets": int(valid.secondary_category.notna().sum()),
        "repeat": {
            "said_repeat_complaint": int(sum(is_repeat_complaint(clean_message(t).text) for t in valid.message_text)),
            "customers_with_2plus_tickets": int((cust >= 2).sum()),
        },
        "model_version": valid.model_version.mode().iloc[0] if n else "",
    }
    facts["monitoring"] = signals(df, categories)
    facts["emerging_issues"] = emerging_issues(valid)
    return facts
