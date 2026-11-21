"""The JSON payloads behind the web UI (app.py), kept here so they can be tested without a server.

Nothing in this module writes to disk: an uploaded CSV is classified in memory and only the results
go back to the browser.
"""
import io
import time
from collections import Counter

import pandas as pd

from .chat.kb import FACTS
from .chat.respond import store_name
from .config import settings
from .pipeline import explain_text, process_frame
from .replies import suggest_reply

MAX_BATCH_ROWS = 5000
SAMPLES = {  # whitelisted demo files the UI can classify without an upload
    "week1": "tickets_2026-07-01_to_2026-07-07.csv",
    "week2": "tickets_2026-07-08_to_2026-07-14.csv",
}


class BadRequest(ValueError):
    """A problem with the user's input, safe to show in the UI."""


def _names(code: str | None) -> dict | None:
    return {"code": code, "ar": store_name(code, "ar"), "en": store_name(code, "en")} if code else None


def classify_one(text: str, backend, llm=None) -> dict:
    text = str(text or "")[: settings.max_message_chars]
    record, details = explain_text(text, backend)
    reply, source, _ = suggest_reply(record, text, llm)
    rec = record.model_dump(mode="json")
    details["store"] = _names(record.store_mentioned)
    return {"record": rec, "details": details, "reply": reply, "reply_source": source}


def _read_csv(csv_text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(csv_text), dtype=str, on_bad_lines="skip").fillna("")
    except Exception as e:  # malformed file
        raise BadRequest(f"could not read the CSV ({type(e).__name__})") from None
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    text_col = next((c for c in ("message_text", "text", "message", "complaint") if c in df.columns), None)
    if text_col is None:
        raise BadRequest("the CSV needs a 'message_text' column")
    if len(df) > MAX_BATCH_ROWS:
        raise BadRequest(f"too many rows ({len(df)}); the limit is {MAX_BATCH_ROWS}")
    if "ticket_id" not in df.columns or not df.ticket_id.str.strip().any():
        df["ticket_id"] = [f"R{i + 1:05d}" for i in range(len(df))]
    df["ticket_id"] = df.ticket_id.str.strip()
    df["message_text"] = df[text_col]
    return df.drop_duplicates("ticket_id").reset_index(drop=True)  # same rule as the weekly job


def classify_batch(csv_text: str | None, backend, sample: str | None = None) -> dict:
    if sample:
        if sample not in SAMPLES:
            raise BadRequest("unknown sample")
        csv_text = (settings.incoming_dir / SAMPLES[sample]).read_text(encoding="utf-8-sig")
    if not csv_text or not csv_text.strip():
        raise BadRequest("the file is empty")
    t0 = time.perf_counter()
    df = _read_csv(csv_text)
    records = process_frame(df, backend)
    rows = []
    for r, text in zip(records, df.message_text):
        rows.append({"ticket_id": r.ticket_id, "category": r.category.value,
                     "secondary_category": r.secondary_category.value if r.secondary_category else "",
                     "product_mentioned": r.product_mentioned or "", "store_mentioned": r.store_mentioned or "",
                     "sentiment": r.sentiment.value, "urgency": r.urgency, "is_valid": r.is_valid,
                     "injection_flag": r.injection_flag, "confidence": round(r.confidence, 3),
                     "text": str(text)[:300]})
    valid = [r for r in rows if r["is_valid"]]
    stores = Counter(r["store_mentioned"] for r in valid if r["store_mentioned"])
    products = Counter(r["product_mentioned"] for r in valid if r["product_mentioned"])
    summary = {
        "rows_received": int(len(df)), "valid": len(valid), "invalid": len(rows) - len(valid),
        "injection_flagged": sum(r["injection_flag"] for r in rows),
        "urgent": sum(r["urgency"] == 3 for r in valid),
        "by_category": dict(Counter(r["category"] for r in valid).most_common()),
        "secondary": dict(Counter(r["secondary_category"] for r in valid if r["secondary_category"]).most_common()),
        "by_urgency": {str(u): sum(r["urgency"] == u for r in valid) for u in (1, 2, 3)},
        "top_stores": [{**_names(code), "count": n} for code, n in stores.most_common(5)],
        "top_products": [{"brand": b, "count": n} for b, n in products.most_common(5)],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        "model_version": records[0].model_version if records else "",
    }
    return {"summary": summary, "rows": rows, "sample": sample}


def chat_payload(state, result) -> dict:
    lang = result.plan.lang
    return {"session_id": state.session_id, "reply": result.reply, "source": result.source, "action": result.plan.action,
            "facts": result.plan.fact_ids(),
            "facts_text": [{"id": f, "text": FACTS[f].text(lang)} for f in dict.fromkeys(result.plan.fact_ids())],
            "latency_ms": result.latency_ms, "state": result.state}


def demo_customers(orders) -> dict:
    by_customer: dict[str, list] = {}
    for o in sorted(orders.all_orders(), key=lambda o: o.order_id):
        by_customer.setdefault(o.customer_id, []).append({
            "order_id": o.order_id, "governorate": o.governorate, "promised": o.promised.isoformat(),
            "delivered": o.delivered.isoformat() if o.delivered else None, "status": o.status,
            "payment": o.payment, "items": list(o.items)})
    return {"today": orders.today.isoformat(), "note": "DEMO DATA: fictional orders",
            "customers": [{"customer_id": c, "orders": v} for c, v in sorted(by_customer.items())]}


def health(backend, llm) -> dict:
    return {"classifier": getattr(backend, "name", "?"), "llm": getattr(llm, "name", None),
            "chat_llm": bool(llm) and settings.chat_use_llm, "chat_phrasing": settings.chat_phrasing}
