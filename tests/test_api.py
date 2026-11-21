"""The web UI's payloads (orvana/api.py): same records as the pipeline, safe CSV handling, no surprises."""
import pandas as pd
import pytest

from orvana import api
from orvana.backends.tfidf import TfidfBackend
from orvana.chat.manager import ChatBot
from orvana.chat.orders import DemoOrderService
from orvana.config import settings
from orvana.pipeline import explain_text, process_text


@pytest.fixture(scope="module")
def tfidf():
    return TfidfBackend.load()


def test_explain_gives_exactly_the_pipeline_record_on_the_gold_texts(tfidf):
    gold = pd.read_csv(settings.data_dir / "tickets_dev_gold.csv", dtype=str).fillna("")
    for text in gold.message_text:
        rec, details = explain_text(text, tfidf)
        assert rec == process_text(text, tfidf)
        if rec.is_valid:
            assert details["segments"][0]["category"] == rec.category.value
            assert len(details["segments"]) == (2 if rec.secondary_category else 1) or rec.secondary_category is None


def test_explanation_shows_the_split_and_the_urgency_reason(tfidf):
    out = api.classify_one("الشاحن VoltX بيسخن جامد وريحته شياط وكمان الكاشير في فرع الدقي اتعامل وحش", tfidf)
    d = out["details"]
    assert [s["role"] for s in d["segments"]] == ["primary", "secondary"]
    assert out["record"]["urgency"] == 3 and d["flags"]["safety"] and "safety" in d["urgency_reason"]
    assert d["store"] == {"code": "GIZ-02", "ar": "فرع الدقي", "en": "Dokki branch"}
    assert out["reply"]


def test_batch_needs_a_text_column(tfidf):
    with pytest.raises(api.BadRequest, match="message_text"):
        api.classify_batch("id,body\n1,hello\n", tfidf)
    with pytest.raises(api.BadRequest, match="empty"):
        api.classify_batch("   ", tfidf)


def test_batch_generates_ids_and_drops_duplicates(tfidf):
    out = api.classify_batch("message_text\nالطلب اتأخر\nالنقط ما نزلتش\n", tfidf)
    assert [r["ticket_id"] for r in out["rows"]] == ["R00001", "R00002"]
    out = api.classify_batch("ticket_id,message_text\nA,الطلب اتأخر\nA,الطلب اتأخر\nB,\n", tfidf)
    assert out["summary"]["rows_received"] == 2 and out["summary"]["invalid"] == 1


def test_batch_row_cap(tfidf, monkeypatch):
    monkeypatch.setattr(api, "MAX_BATCH_ROWS", 3)
    with pytest.raises(api.BadRequest, match="too many rows"):
        api.classify_batch("message_text\na\nb\nc\nd\n", tfidf)


def test_batch_summary_matches_the_rows_on_a_real_week(tfidf):
    out = api.classify_batch(None, tfidf, sample="week1")
    s, rows = out["summary"], out["rows"]
    valid = [r for r in rows if r["is_valid"]]
    assert s["rows_received"] == len(rows) == 320
    assert sum(s["by_category"].values()) == s["valid"] == len(valid)
    assert sum(s["by_urgency"].values()) == s["valid"] and s["urgent"] == s["by_urgency"]["3"]
    with pytest.raises(api.BadRequest):
        api.classify_batch(None, tfidf, sample="../../etc/passwd")


def test_chat_payload_gives_fact_texts_in_the_conversation_language(tfidf):
    bot = ChatBot(tfidf, llm=None, orders=DemoOrderService())
    st = bot.new_state(None)
    out = api.chat_payload(st, bot.turn(st, "do points expire?"))
    assert out["facts_text"] and all(f["text"].isascii() for f in out["facts_text"])


def test_demo_customers_lists_the_fixture():
    out = api.demo_customers(DemoOrderService())
    assert out["today"] == "2026-07-15" and "DEMO" in out["note"]
    assert {c["customer_id"] for c in out["customers"]} == {"DEMO-1", "DEMO-2", "DEMO-3"}
