"""The Arabic summary: numbers must exist AND be attached to the right thing; first weeks have no
'change vs last week'; a new incident must be mentioned."""
from orvana.report.summary_llm import consistency_problems, llm_facts, template_summary

FACTS = {
    "period": {"start": "2026-07-01", "end": "2026-07-07"},
    "volume": {"valid_tickets": 320, "prev_valid_tickets": None, "change_vs_prev_pct": None},
    "categories": [{"category": "delivery_delay", "count": 74, "share_pct": 23.1},
                   {"category": "product_quality", "count": 44, "share_pct": 13.8},
                   {"category": "app_issue", "count": 33, "share_pct": 10.3}],
    "urgency": {"3": 44}, "urgent_tickets": [], "staff_behavior_by_store": [], "top_stores": [],
    "emerging_issues": [], "safety": {"confirmed": 12, "to_confirm": 3, "second_issue": 2},
}


def test_wrong_number_for_a_category_is_caught():
    bad = "أكبر فئتين هما تأخير التوصيل (74 شكوى) ومشاكل الأبلكيشن (44 شكوى)."
    assert any("مشاكل الأبلكيشن" in p for p in consistency_problems(bad, FACTS))
    good = "أكبر فئتين هما تأخير التوصيل (74 شكوى) وجودة المنتج (44 شكوى)."
    assert consistency_problems(good, FACTS) == []


def test_first_week_has_no_previous_week_fields():
    data = llm_facts(FACTS)
    assert "عدد_الأسبوع_السابق" not in data and "ملاحظة" in data


def test_new_incident_must_be_mentioned():
    facts = {**FACTS, "emerging_issues": [{"terms": ["update", "5.2"], "tickets": 62, "share_pct": 19.1,
                                           "categories": {"app_issue": 38, "refund_request": 24}}]}
    assert consistency_problems("استقبلنا 320 شكوى.", facts)
    assert consistency_problems("مشكلة جديدة: 62 شكوى عن تحديث 5.2 في مشاكل الأبلكيشن 62.", facts) == []


def test_template_is_always_consistent():
    assert consistency_problems(template_summary(FACTS), FACTS) == []


def test_safety_counts_are_separate_from_urgent_and_verified():
    from orvana.report.validate_numbers import check_summary
    text = template_summary(FACTS)
    # a safety report can be the 2nd problem of an urgency-2 ticket, so it is not "of which" (منها) the urgent ones
    assert "منها" not in text and "12 حالة سلامة" in text and "3 بلاغ سلامة يحتاج تأكيد" in text
    assert check_summary(text, FACTS) == []
    data = llm_facts(FACTS)
    assert data["حالات_سلامة_منتج_تحتاج_تصعيد_خلال_ساعتين"] == 12 and data["بلاغات_سلامة_تحتاج_تأكيد_من_العميل_أولا"] == 3
