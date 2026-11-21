"""Urgency and sentiment rules, inferred from the dev gold set (documented in docs/URGENCY_RULES.md).

Urgency depends on the PRIMARY problem plus whether the customer says this is a repeat complaint:
  3  repeat complaint ('third time'), OR primary = product_quality with a safety issue
     (expired / overheating / burnt smell), OR primary = refund_request for a large double charge
  1  primary = inquiry
  2  everything else
"""
from .config import settings
from .extractors import double_charge_amount, has_safety_issue, is_repeat_complaint
from .schema import Category, Sentiment


def urgency(primary: Category, primary_segment: str, full_text: str) -> int:
    if is_repeat_complaint(full_text):
        return 3
    if primary == Category.product_quality and has_safety_issue(primary_segment):
        return 3
    if primary == Category.refund_request:
        amount = double_charge_amount(primary_segment)
        if amount is not None and amount >= settings.double_charge_urgent_egp:
            return 3
    if primary == Category.inquiry:
        return 1
    return 2


def sentiment(primary: Category, model_sentiment: Sentiment | None = None) -> Sentiment:
    """A complaint is negative even when phrased as thanks (sarcasm: 'thanks for the super fast
    delivery that took a week 👏'). Only inquiries can be neutral/positive."""
    if primary != Category.inquiry:
        return Sentiment.negative
    if model_sentiment == Sentiment.positive:
        return Sentiment.positive
    return Sentiment.neutral
