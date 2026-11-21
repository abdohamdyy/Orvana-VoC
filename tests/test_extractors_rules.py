"""The deterministic parts must reproduce the gold exactly (product, store, secondary presence,
urgency and sentiment given the true category)."""
import pandas as pd
import pytest

from orvana.config import settings
from orvana.extractors import find_product, find_store, split_segments
from orvana.normalize import clean_message
from orvana.rules import sentiment, urgency
from orvana.schema import Category

GOLD = pd.read_csv(settings.data_dir / "tickets_dev_gold.csv")
none = lambda v: None if pd.isna(v) else v  # noqa: E731


@pytest.mark.parametrize("row", list(GOLD.itertuples()), ids=lambda r: r.ticket_id)
def test_gold_row(row):
    c = clean_message(row.message_text)
    segs = split_segments(c.text)
    cat = Category(row.category)
    assert find_product(c.text) == none(row.product_mentioned)
    assert find_store(c.text) == none(row.store_mentioned)
    assert (len(segs) == 2) == (not pd.isna(row.secondary_category))
    assert urgency(cat, segs[0], c.text) == row.urgency
    assert sentiment(cat).value == row.sentiment


@pytest.mark.parametrize("text,expected", [
    ("mawa3eed DLT-088 eh yom el gom3a?", "DLT-08"),          # duplicated digit in a code
    ("هو فيه توصيل لـفرع طنطا؟", "DLT-01"),                   # glued prefix + Arabic question mark
    ("السعر في فرع المنصورة 2 غلط", "DLT-06"),                 # longest name wins over 'المنصورة'
    ("el se3r 3al raf f far3Downtown gher el fatora", "CAI-07"),  # glued Franco
    ("law sama7t fe tawseel l far3 Soag?", "UPE-03"),          # dropped letter (fuzzy)
    ("no store here", None),
])
def test_store_edge_cases(text, expected):
    assert find_store(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("مولتو هيرجع امتى؟", "Edita"), ("el Moltohayerga3 emta?", "Edita"), ("موبايل ساامسونج باظ", "Samsung"),
    ("my Orvana Plus points disappeared", None),  # loyalty programme, not a product
])
def test_product_edge_cases(text, expected):
    assert find_product(text) == expected


def test_double_charge_threshold():
    small = "عايز استرد 850 جنيه اتخصموا مرتين من الكارت"
    big = "عايز استرد 1900 جنيه اتخصموا مرتين من الكارت"
    assert urgency(Category.refund_request, small, small) == 2
    assert urgency(Category.refund_request, big, big) == 3


def test_sarcasm_is_negative():
    assert sentiment(Category.delivery_delay).value == "negative"
