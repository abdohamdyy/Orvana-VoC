"""Complaint text -> TicketRecord.

hybrid (default): clean -> split into [primary, secondary] segments -> backend picks a category per
                  segment -> dictionaries find product/store -> rules give urgency/sentiment.
llm_only (Stage 2 v1 baseline): the LLM fills every field from the raw text; we only whitelist values.
"""
import pandas as pd

from .extractors import find_product, find_store, load_catalog, split_segments
from .normalize import CleanMessage, clean_message
from .rules import sentiment, urgency
from .schema import Category, Sentiment, TicketRecord


def _invalid(ticket_id: str, c: CleanMessage, version: str) -> TicketRecord:
    """Empty message or pure prompt-injection: kept for audit, excluded from report statistics."""
    return TicketRecord(ticket_id=ticket_id, category=Category.inquiry, sentiment=Sentiment.neutral, urgency=1,
                        is_valid=False, injection_flag=c.injection_flag, confidence=0.0, model_version=version)


def _record(tid: str, c: CleanMessage, segs: list[str], p: list, version: str) -> TicketRecord:
    """One record from the segments and their (category, confidence) predictions."""
    if not segs:
        return _invalid(tid, c, version)
    primary = p[0][0]
    return TicketRecord(
        ticket_id=tid,
        category=primary,
        secondary_category=p[1][0] if len(p) > 1 else None,
        product_mentioned=find_product(c.text),
        store_mentioned=find_store(c.text),
        sentiment=sentiment(primary),
        urgency=urgency(primary, segs[0], c.text),
        injection_flag=c.injection_flag,
        confidence=min(conf for _, conf in p),
        model_version=version,
    )


def _hybrid(ids: list[str], cleaned: list[CleanMessage], backend) -> list[TicketRecord]:
    segs_per_ticket = [split_segments(c.text) if c.is_valid else [] for c in cleaned]
    flat = [s for segs in segs_per_ticket for s in segs]
    preds = backend.classify(flat) if flat else []

    records, k = [], 0
    for tid, c, segs in zip(ids, cleaned, segs_per_ticket):
        p, k = preds[k : k + len(segs)], k + len(segs)
        records.append(_record(tid, c, segs, p, backend.name))
    return records


def explain_text(text: str, backend, ticket_id: str = "adhoc") -> tuple[TicketRecord, dict]:
    """The record (built exactly as process_text builds it) plus how it was reached, for the UI."""
    from .extractors import double_charge_amount, has_safety_issue, is_repeat_complaint
    from .report.stats import safety_report, urgency_reason

    c = clean_message(text)
    segs = split_segments(c.text) if c.is_valid else []
    p = backend.classify(segs) if segs else []
    rec = _record(ticket_id, c, segs, p, backend.name)
    details = {
        "cleaned_text": c.text,
        "injection_removed": c.injection_flag,
        "is_valid": c.is_valid,
        "segments": [{"text": s, "role": "primary" if i == 0 else "secondary", "category": cat.value,
                      "confidence": round(float(conf), 3)} for i, (s, (cat, conf)) in enumerate(zip(segs, p))],
        "flags": {"safety": bool(segs) and has_safety_issue(segs[0]), "repeat": is_repeat_complaint(c.text),
                  "double_charge_egp": double_charge_amount(segs[0]) if segs else None,
                  # where the hazard is and whether the product type makes it plausible (Quality team routing)
                  "safety_report": safety_report(text, rec.product_mentioned) if rec.is_valid else None},
        "urgency_reason": urgency_reason(text, rec.category.value, rec.product_mentioned) if rec.is_valid else "",
    }
    return rec, details


def _llm_only(ids: list[str], raw: list[str], cleaned: list[CleanMessage], backend) -> list[TicketRecord]:
    cat = load_catalog()
    brands = {b for _, b in cat.brand_aliases}
    records = []
    for tid, text, c in zip(ids, raw, cleaned):
        if not str(text or "").strip():
            records.append(_invalid(tid, c, backend.name))
            continue
        out = backend.extract_all(str(text))
        prod = next((b for b in brands if b.lower() == str(out.get("product_mentioned", "")).lower()), None)
        store = str(out.get("store_mentioned", "")).upper()
        sec = out.get("secondary_category")
        records.append(TicketRecord(
            ticket_id=tid,
            category=out["category"],
            secondary_category=sec if sec in Category.__members__ else None,
            product_mentioned=prod,
            store_mentioned=store if store in cat.store_ids else None,
            sentiment=out.get("sentiment", "negative"),
            urgency=out.get("urgency") if out.get("urgency") in (1, 2, 3) else 2,
            injection_flag=c.injection_flag,
            model_version=backend.name,
        ))
    return records


def process_frame(df: pd.DataFrame, backend, mode: str = "hybrid") -> list[TicketRecord]:
    """df needs columns ticket_id, message_text (other columns are ignored)."""
    ids = df.ticket_id.astype(str).tolist()
    raw = df.message_text.tolist()
    cleaned = [clean_message(t) for t in raw]
    if mode == "llm_only":
        return _llm_only(ids, raw, cleaned, backend)
    return _hybrid(ids, cleaned, backend)


def process_text(text: str, backend, ticket_id: str = "adhoc") -> TicketRecord:
    return process_frame(pd.DataFrame({"ticket_id": [ticket_id], "message_text": [text]}), backend)[0]
