"""B1: suggested replies must respect the CS policy; the rule checker must catch violations."""
import pytest

from orvana.replies import check_reply, suggest_reply, template_reply
from orvana.schema import Category, Sentiment, TicketRecord


def rec(cat, urgency=2, **kw):
    return TicketRecord(ticket_id="T", category=Category(cat), sentiment=Sentiment.negative, urgency=urgency, **kw)


SAFETY_TEXT = "زيت Orvana Select بيسخن جامد وريحته شياط"


def test_cash_promise_rejected():
    assert "mentions cash compensation" in check_reply("هنعوضك بفلوس كاش فوراً يا فندم، بنعتذر جداً", rec("delivery_delay"), "الأوردر متأخر")


def test_invented_number_rejected():
    probs = check_reply("هنرجعلك الفلوس خلال 2 يوم يا فندم، بنعتذر جداً عن التأخير", rec("refund_request"), "رجعت المنتج")
    assert any("number 2" in p for p in probs)


def test_policy_number_accepted():
    assert check_reply("بنعتذر جداً، الفلوس بترجع على الكارت خلال 5 أيام عمل.", rec("refund_request"), "رجعت المنتج") == []


def test_safety_needs_quality_escalation():
    r = rec("product_quality", urgency=3)
    assert "safety case without Quality-team escalation" in check_reply("بنعتذر جداً يا فندم وهنتابع معاك", r, SAFETY_TEXT)
    assert check_reply("بنعتذر جداً، الشكوى اتصعدت لفريق الجودة والمنتج هيتسحب من الفرع.", r, SAFETY_TEXT) == []


@pytest.mark.parametrize("cat", [c.value for c in Category])
def test_template_always_passes_checks(cat):
    text = SAFETY_TEXT if cat == "product_quality" else "عندي مشكلة"
    r = rec(cat)
    assert check_reply(template_reply(r, text), r, text) == []


@pytest.mark.parametrize("text,voucher", [("الأوردر متأخر بقاله 7 أيام", "50"), ("my order is 10 days late", "100"),
                                          ("el order et2a5ar 2 ayam", None)])
def test_voucher_decided_by_code(text, voucher):
    from orvana.replies import delivery_line
    line = delivery_line(text)
    assert (voucher in line) if voucher else ("قسيمة" not in line)


def test_franco_digits_are_not_numbers():
    r = rec("app_issue")
    assert any("number 3" in p for p in check_reply("بنعتذر جداً، المشكلة هتتحل خلال 3 أيام", r, "mesh 3aref a3mel login men embare7"))


def test_invalid_record_gets_clarifying_question():
    r = TicketRecord(ticket_id="T", category=Category.inquiry, sentiment=Sentiment.neutral, urgency=1, is_valid=False)
    reply, source, _ = suggest_reply(r, "", None)
    assert source == "template" and "توضح" in reply
